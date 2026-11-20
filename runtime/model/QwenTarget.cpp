#include "model/QwenTarget.hpp"

#include "StderrLine.hpp"
#include "model/Flint.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/WeightStore.hpp"
#include "ops/AneFfn.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"
#include "ops/PageStorage.hpp"
#include "ops/RowCopy.hpp"
#include "ops/WarmupTrace.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

// A prefill sequence binds every GDN layer's state cells, in and out, and so
// does each state it captures; with auxiliary state, its cells too.
void requirePrefillStates(const QwenTargetGeometry &geometry, const QwenTargetPrefillSequence &sequence) {
  const auto everyLayer = [&](std::span<const metal::MetalBuffer> views) {
    return views.size() == geometry.stateLayout.layers;
  };
  const bool auxiliary = geometry.stateLayout.auxiliaryBytes != 0;
  if (!everyLayer(sequence.convolutionIn) || !everyLayer(sequence.convolutionOut) ||
      !everyLayer(sequence.recurrentIn) || !everyLayer(sequence.recurrentOut) ||
      (auxiliary && (!sequence.auxiliaryIn || !sequence.auxiliaryOut)) ||
      !std::ranges::all_of(sequence.stateCaptures, [&](const QwenTargetStateCapture &capture) {
        return everyLayer(capture.convolution) && everyLayer(capture.recurrent) && (capture.auxiliary || !auxiliary);
      }) ||
      (!sequence.endCodes.empty() && !everyLayer(sequence.endCodes)))
    throw std::invalid_argument("Qwen prefill state layer mismatch");
}

} // namespace

void QwenTargetGeometry::finish(std::span<const ops::BlockMoeWeights *const> moeBlocks,
                                const ops::Projection &logits) {
  moeRouterFormat = ops::moeRouterFormat(moeBlocks);
  prefillProjections = decodeProjections;
  include(logits);
  // The draft's head over its subset of the vocabulary: a view of the head's
  // rows (DFlashDraft), the head's shape over fewer of them.
  ops::ProjectionShape subset = logits.shape();
  subset.outputSize = draftVocabulary.columns(vocabularySize);
  decodeProjections.push_back(subset);
  for (auto *shapes : {&prefillProjections, &decodeProjections, &gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
}

void includeDenseFfn(QwenTargetGeometry &geometry, ops::SwiGluProjections ffn) {
  const ops::ProjectionShape gate = ffn.gate->shape(), up = ffn.up->shape();
  if (gate.outputSize != up.outputSize || gate.inputSize != up.inputSize || gate.rotated != up.rotated)
    throw WeightStoreError("fused gate/up projections must have matching shapes");
  geometry.include(*ffn.gate);
  geometry.include(*ffn.up);
  geometry.include(*ffn.down);
  geometry.gateUpProjections.push_back(up);
}

QwenTarget::QwenTarget(const QwenTargetWeightsBase &weights, const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : weightsBase_(weights), geometry_(geometry), backend_(backend), operators_(operators) {
  if (!geometry_.valid()) throw std::invalid_argument("Qwen target weights do not match execution geometry");
  if (!ops::PagedAttention::compiles(geometry_.attentionQueryHeads, geometry_.kvLayout, geometry_.attentionGate,
                                     geometry_.rotaryPairs))
    throw std::invalid_argument("no attention kernel for the target's output gate and RoPE");
}

struct QwenTarget::WarmupTrace {
  struct Point { const char *stage; uint32_t layer, width, routes; size_t after; };
  static constexpr uint32_t capacity = 1024;
  const char *phase;
  bool request = false;
  metal::MetalBuffer results, laneRows;
  uint32_t layer = 0;
  std::vector<Point> points;
};

QwenTarget::~QwenTarget() = default;

void QwenTarget::beginWarmupTrace(const char *phase) {
  warmupTrace_ = std::make_unique<WarmupTrace>();
  auto &trace = *warmupTrace_;
  trace.phase = phase;
  trace.results = backend_.allocateBuffer(WarmupTrace::capacity * sizeof(uint32_t),
                                          metal::BufferStorage::Shared, "B1 warmup nonfinite locations");
  std::fill_n(static_cast<uint32_t *>(trace.results.contents()), WarmupTrace::capacity, UINT32_MAX);
}

void QwenTarget::observe(metal::CommandGraph &graph, const char *stage, const metal::MetalBuffer &buffer,
                         uint32_t rows, uint32_t width, uint32_t bytes, uint32_t routes,
                         const metal::MetalBuffer &routeRows, uint32_t stride) const {
  if (!warmupTrace_) return;
  auto &trace = *warmupTrace_;
  if (!stride) stride = width;
  const uint64_t count = uint64_t{rows} * routes * width;
  const uint64_t extent = (uint64_t{rows} * routes - 1) * stride + width;
  const auto slot = static_cast<uint32_t>(trace.points.size());
  if (!count || count > UINT32_MAX || slot >= WarmupTrace::capacity || stride < width || extent * bytes > buffer.sizeBytes() ||
      (routeRows && uint64_t{rows} * routes * sizeof(uint32_t) > routeRows.sizeBytes()))
    throw std::logic_error("invalid B1 warmup trace extent");
  trace.points.push_back({stage, trace.layer, width, routes, graph.dispatches().size()});
  ops::addWarmupTrace(graph, buffer, trace.results, trace.laneRows, routeRows, rows, width, routes, bytes, slot, stride);
}

void QwenTarget::finishWarmupTrace() {
  if (!warmupTrace_) return;
  const auto trace = std::move(warmupTrace_);
  const auto *first = static_cast<const uint32_t *>(trace->results.contents());
  for (size_t i = 0; i < trace->points.size(); ++i) {
    if (first[i] == UINT32_MAX) continue;
    const auto &p = trace->points[i];
    logLine("NAN_TRACE phase=", trace->phase, " layer=", p.layer, " stage=", p.stage, " checkpoint=", i,
            " row=", first[i] / p.width / p.routes, " route=", first[i] / p.width % p.routes,
            " element=", first[i] % p.width);
    return;
  }
  logLine("NAN_TRACE phase=", trace->phase, " finite_checkpoints=", trace->points.size());
}

void QwenTarget::beginRequestTrace() {
  beginWarmupTrace("cold-prefill");
  warmupTrace_->request = true;
}

void QwenTarget::traceFinalRow(metal::CommandGraph &graph, const char *stage, const metal::MetalBuffer &row) const {
  observe(graph, stage, row, 1, geometry_.hiddenSize);
}

std::function<void()> QwenTarget::takeRequestTrace(const metal::CommandGraph &graph, uint64_t request,
                                                    uint32_t rows, QwenPrefillLayers layers) {
  // The ticket owns the result and a copy of the dispatch descriptions until its command finishes.
  std::shared_ptr<WarmupTrace> trace(std::move(warmupTrace_));
  std::vector<std::string> dispatches;
  for (const auto &d : graph.dispatches()) {
    std::ostringstream line;
    if (d.pipelineName != "warmup_first_nonfinite") {
      line << d.pipelineName << (d.indirectThreadgroups.buffer ? " indirect_bound=" : " grid=")
           << d.threadgroups.x << 'x' << d.threadgroups.y << 'x' << d.threadgroups.z
           << " threads=" << d.threadsPerThreadgroup.x << 'x' << d.threadsPerThreadgroup.y << 'x' << d.threadsPerThreadgroup.z;
      for (const auto constant : d.constants) line << " fc=" << constant;
    }
    dispatches.push_back(line.str());
  }
  return [trace = std::move(trace), dispatches = std::move(dispatches), request, rows, layers]() {
    const auto *first = static_cast<const uint32_t *>(trace->results.contents());
    for (size_t i = 0; i < trace->points.size(); ++i) {
      if (first[i] == UINT32_MAX) continue;
      const auto &point = trace->points[i];
      logLine("PREFILL_TRACE request=", request, " rows=", rows, " layers=[", layers.first, ",", layers.end,
              ") layer=", point.layer, " tensor=", point.stage, " checkpoint=", i, " row=",
              first[i] / point.width / point.routes, " route=", first[i] / point.width % point.routes,
              " element=", first[i] % point.width, " graph_position=", point.after);
      size_t start = 0;
      for (size_t j = 0; j < i; ++j)
        if (trace->points[j].layer < point.layer) start = trace->points[j].after;
      for (size_t j = start; j < point.after; ++j)
        if (!dispatches[j].empty()) logLine("PREFILL_DISPATCH request=", request, " i=", j, " ", dispatches[j]);
      return;
    }
    logLine("PREFILL_TRACE request=", request, " rows=", rows, " layers=[", layers.first, ",", layers.end,
            ") finite_checkpoints=", trace->points.size());
  };
}

void QwenTarget::discardWarmupTrace() { warmupTrace_.reset(); }

QwenTargetScratchBytes QwenTarget::allocatedScratchBytes() const noexcept { return {}; }

const ops::Projection &QwenTarget::vocabularyProjection() const noexcept {
  return weightsBase_.logitsProjection;
}

uint32_t QwenTarget::headStorageLanes(uint32_t lanes) const {
  constexpr uint32_t laneRows = ExecutionLimits::draftQueryRows;
  return std::max(lanes * laneRows,
                  operators_.linear().storageRows(lanes * laneRows, vocabularyProjection().shape())) /
         laneRows;
}

void QwenTarget::addInsideCapture(metal::CommandGraph &, metal::MetalBuffer, metal::MetalBuffer) const {}

void QwenTarget::prepareVerifyLane(uint32_t, uint64_t, const QwenTargetRequest &) {}

std::vector<uint32_t> QwenTarget::verifyTableRows(const ops::LaneRows &) const { return {}; }

void QwenTarget::addVerifyInput(metal::CommandGraph &, const QwenVerifyInput &) const {}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &, QwenTargetCommitBuffers buffers,
                                const ops::StepRows &step) const {
  if (!step.lanes || step.lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  if (geometry_.stateLayout.auxiliaryBytes)
    for (uint32_t lane = 0; lane < step.lanes; ++lane)
      if (!buffers.auxiliaryIn[lane] || !buffers.auxiliaryOut[lane])
        throw std::invalid_argument("Qwen state commit lane without its auxiliary state");
}

void QwenTarget::addStateFold(metal::CommandGraph &graph, QwenTargetFoldBuffers buffers, uint32_t rows) const {
  if (geometry_.stateLayout.auxiliaryBytes)
    throw std::logic_error("a target with auxiliary state folds no state in decode");
  if (!geometry_.stateLayout.layers)
    return;
  ops::GDN::addFold(graph,
                    {std::move(buffers.state), std::move(buffers.gdnTape), buffers.stepSlot,
                     std::move(buffers.retainedRows), std::move(buffers.fold)},
                    geometry_.gdnShape(), geometry_.stateLayout.layers, rows, geometry_.stateLayout.strides(),
                    geometry_.stateLayout.cachedRecurrentLayerBytes(), geometry_.stateLayout.cachedCodes);
}

void QwenTarget::requireLayers(size_t layers, size_t attentionLayers) const {
  if (layers != geometry_.layers || attentionLayers != geometry_.kvLayout.attentionLayers)
    throw std::invalid_argument("Qwen target weights do not match execution geometry");
}

void QwenTarget::requireLayerPartition(uint32_t gdnLayers, uint32_t attentionLayers) const {
  if (gdnLayers != geometry_.stateLayout.layers || attentionLayers != geometry_.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
}

void QwenTarget::requirePrefill(std::span<const QwenTargetPrefillChunk> chunks, const kv::PageStorage &kv,
                                QwenPrefillLayers layers, const ops::AneFfn *aneFfn) const {
  // The Neural Engine splits a dense target's commands, which are one chunk
  // and run whole.
  const bool whole = !layers.first && layers.end == geometry_.layers;
  if (chunks.empty() || kv.layers().size() != geometry_.kvLayout.attentionLayers ||
      (aneFfn && (chunks.size() > 1 || !whole)) || layers.first >= layers.end || layers.end > geometry_.layers)
    throw std::invalid_argument("invalid Qwen prefill");
}

QwenTarget::PrefillStep QwenTarget::prefillStep(metal::CommandGraph &graph, const QwenTargetPrefillChunk &chunk,
                                                const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                ops::AneFfn *aneFfn) const {
  if (chunk.sequences.empty() || chunk.sequences.size() > ExecutionLimits::maximumBatchWidth || !chunk.rows ||
      chunk.rows > ExecutionLimits::prefillTokenBudget)
    throw std::invalid_argument("invalid Qwen prefill chunk");
  PrefillStep step{graph, chunk.buffers, chunk.sequences, chunk.rows, kv.layers(), spilled};
  step.decode = chunk.decode;
  if (step.decode && (step.decode->step.lanes != chunk.sequences.size() || step.decode->step.rows != chunk.rows))
    throw std::invalid_argument("a Qwen decode step's lanes or rows do not match its chunk");
  for (size_t index = 0; index < chunk.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = chunk.sequences[index];
    // A decode lane's states are the step's (QwenTargetDecodeBuffers).
    if (!step.decode) requirePrefillStates(geometry_, sequence);
    else if (geometry_.stateLayout.auxiliaryBytes && !sequence.auxiliaryIn)
      throw std::invalid_argument("Qwen decode lane without its auxiliary state");
    step.chunks[index] = sequence.chunk;
    step.pageTables[index] = sequence.pageTable;
  }
  const ops::AttentionWorkspace workspace{chunk.buffers.attentionPartials.sizeBytes(),
                                          chunk.buffers.attentionStatistics.sizeBytes()};
  step.attention.emplace(operators_.attention(step.attentionSequences(), geometry_.attentionQueryHeads,
                                              geometry_.kvLayout, workspace));
  if (const uint32_t window = geometry_.kvLayout.windowTokens)
    step.windowedAttention.emplace(operators_.attention(step.attentionSequences(), geometry_.attentionQueryHeads,
                                                        geometry_.kvLayout, workspace, {.windowTokens = window}));
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe)
    step.moe = operators_.moe(geometry_.moeBlockShape(), geometry_.moeRouterFormat, chunk.rows);
  if (aneFfn && aneFfn->splits(chunk.rows)) step.aneFfn = aneFfn;
  return step;
}

void QwenTarget::planOutputRows(PrefillStep &step, uint32_t index) const {
  step.readRows.clear();
  // A decode step reads every row's logits.
  if (step.decode || index + 1 != geometry_.layers || !geometry_.isFullAttentionLayer(index) ||
      geometry_.captureSlot(index) || step.sequences.size() == step.rows)
    return;
  for (const QwenTargetPrefillSequence &sequence : step.sequences)
    step.readRows.push_back(sequence.rowBegin + sequence.rows - 1);
  if (step.moe)
    step.moe = operators_.moe(geometry_.moeBlockShape(), geometry_.moeRouterFormat, step.outputRows());
}

void QwenTarget::addReadRows(const PrefillStep &step, const metal::MetalBuffer &compact,
                             const metal::MetalBuffer &output) const {
  const uint32_t width = geometry_.hiddenSize;
  for (uint32_t row = 0; row < step.readRows.size(); ++row)
    ops::RowCopy::add(step.graph, compact, {row, width, 0}, output, {step.readRows[row], width, 0}, 1, width);
}

// The mixer reads the normalized rows as its input projection's plan takes
// them (`normalized`, which the norm wrote), and its output projection reads
// the hidden rows as the decode GDN or the attention gate wrote them for its
// plan. A decode step's GDN scans each lane's rows along its tree from the
// lane's state into its tape (ops::GDN::addDecode); a prompt sequence's scans
// its rows in order from its states into theirs, capturing those inside.
void QwenTarget::addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer, ops::PreparedInput normalized,
                                 BlockOutput output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  linear.add(step.graph,
             {.input = b.normalized, .output = b.gdnPacked, .scratch = b.linearScratch,
              .prepared = std::move(normalized)},
             mixer.inputProjection, linear.plan(mixer.inputProjection, step.rows));
  if (warmupTrace_) {
    const uint32_t live = geometry_.convolutionDimension + geometry_.attentionWidth + 2 * geometry_.gdnValueHeads;
    observe(step.graph, "gdn_packed", b.gdnPacked, step.rows, live, 2, 1, {}, geometry_.packedGdnWidth);
  }
  const ops::LinearPlan outputPlan = linear.plan(
      mixer.outputProjection, step.rows, output.residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None);
  ops::PreparedInput hidden;
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    hidden = ops::GDN::addDecode(
        step.graph,
        {b.gdnPacked, mixer.convolutionWeights, d->currentGdnStates, d->nextGdnStates, d->gdnTape,
         std::span(d->gdnTapeLanes).first(d->step.lanes), b.laneRows, step.rows, d->step.laneCap, mixer.decay,
         mixer.timeBias, mixer.mixerNorm, b.gdnHidden, d->tree, b.linearScratch},
        geometry_.gdnShape(), layer, geometry_.stateLayout.strides(), mixer.outputHeadOrder, outputPlan.input());
  } else {
    for (const QwenTargetPrefillSequence &sequence : step.sequences) {
      const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
        return rowsOf<uint16_t>(buffer, sequence.rowBegin, sequence.rows, width);
      };
      const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
        return rowsOf<float>(buffer, sequence.rowBegin, sequence.rows, width);
      };
      std::vector<ops::GdnPrefillCapture> captures;
      captures.reserve(sequence.stateCaptures.size() + 1);
      for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
        captures.push_back({capture.row, capture.convolution[layer], capture.recurrent[layer]});
      if (!sequence.endCodes.empty())
        captures.push_back({sequence.rows, {}, sequence.endCodes[layer]});
      if (warmupTrace_ && warmupTrace_->request) {
        const auto &state = geometry_.stateLayout;
        observe(step.graph, "gdn_conv_input", sequence.convolutionIn[layer], 1, state.convolutionHistory * state.convolutionChannels);
        observe(step.graph, "gdn_recurrent_input", sequence.recurrentIn[layer], 1,
                state.recurrentGroups * state.recurrentRows * state.recurrentColumns, ops::elementBytes(state.format));
      }
      ops::GDN::addPrefill(
          step.graph, backend_,
          {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights, sequence.convolutionIn[layer],
           sequence.convolutionOut[layer], u16(b.gdnQueries, geometry_.gdnKeyWidth()),
           u16(b.gdnKeys, geometry_.gdnKeyWidth()), u16(b.gdnValues, geometry_.attentionWidth), mixer.decay,
           mixer.timeBias, f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
           sequence.recurrentIn[layer], sequence.recurrentOut[layer], u16(b.recurrent, geometry_.attentionWidth),
           mixer.mixerNorm, u16(b.gdnHidden, geometry_.attentionWidth)},
          geometry_.gdnShape(), sequence.rows, geometry_.stateLayout.format, mixer.outputHeadOrder, captures,
          geometry_.stateLayout.cachedCodes);
      if (warmupTrace_) {
        const auto &state = geometry_.stateLayout;
        if (warmupTrace_->request) {
          observe(step.graph, "gdn_prepare_q", b.gdnQueries, step.rows, geometry_.gdnKeyWidth());
          observe(step.graph, "gdn_prepare_k", b.gdnKeys, step.rows, geometry_.gdnKeyWidth());
          observe(step.graph, "gdn_prepare_v", b.gdnValues, step.rows, geometry_.attentionWidth);
          observe(step.graph, "gdn_prepare_decay", b.gdnDecay, step.rows, geometry_.gdnValueHeads, 4);
          observe(step.graph, "gdn_prepare_beta", b.gdnBeta, step.rows, geometry_.gdnValueHeads);
          observe(step.graph, "gdn_scan_rows", b.recurrent, step.rows, geometry_.attentionWidth);
        }
        observe(step.graph, "seed_gdn_conv_state", sequence.convolutionOut[layer], 1,
                state.convolutionHistory * state.convolutionChannels);
        observe(step.graph, "seed_gdn_recurrent_state", sequence.recurrentOut[layer], 1,
                state.recurrentGroups * state.recurrentRows * state.recurrentColumns, ops::elementBytes(state.format));
      }
    }
  }
  observe(step.graph, "gdn_gated_hidden", b.gdnHidden, step.rows, geometry_.attentionWidth);
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = std::move(output.rows), .residual = std::move(output.residual),
              .scratch = b.linearScratch, .prepared = std::move(hidden)},
             mixer.outputProjection, outputPlan);
}

// A sequence with blocks attends through the lists of its tiles of rows. A
// prompt sequence whose history holds Index pages attends staging, which the
// step's reads of spilled K/V fill with the layer's history and the chunk's
// pages once stored; a decode step whose lanes hold Index pages attends
// through its lists, which those reads locate (SpilledKvReads::verify): its
// rows that read their page tables read only Full pages (SparseKvShape).
void QwenTarget::addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer, ops::PreparedInput normalized,
                                 BlockOutput output, std::span<const ops::AttentionBlocks> blocks) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  const ops::AttentionVariant variant = geometry_.attentionVariant(layer);
  if (!blocks.empty() && blocks.size() != step.sequences.size())
    throw std::invalid_argument("Qwen attention blocks do not match its sequences");
  // A layer that outputs its read rows alone adds its output to their
  // residual rows, which it gathers.
  if (!step.readRows.empty() && !output.residual)
    throw std::logic_error("a Qwen prefill layer that outputs its read rows alone has no residual rows");
  const std::span<const kv::ChunkedPrefillParams> sequences = step.attentionSequences();
  const ops::AttentionRows rows{b.laneRows, step.rows, step.decode ? step.decode->tree : metal::MetalBuffer{}};
  const std::span<const metal::MetalBuffer> pageTables = std::span(step.pageTables).first(sequences.size());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.fullPacked, .scratch = b.linearScratch,
              .prepared = std::move(normalized)},
             mixer.inputProjection, linear.plan(mixer.inputProjection, step.rows));
  observe(step.graph, "attention_qkv_packed", b.fullPacked, step.rows, geometry_.packedFullWidth);
  // The q/k projection stores each row's key and value in its page slot.
  ops::PagedAttention::addProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos, b.ropeSin,
                                     b.fullQueries, step.kvLayers[layer], pageTables, sequences, rows,
                                     geometry_.attentionQueryHeads, geometry_.kvLayout, variant);
  if (warmupTrace_ && warmupTrace_->request)
    observe(step.graph, "attention_rotated_query", b.fullQueries, step.rows, geometry_.attentionWidth);
  std::array<ops::AttentionHistory, ExecutionLimits::maximumBatchWidth> histories{};
  if (step.decode) {
    std::array<ops::AttentionBlocks, ExecutionLimits::maximumBatchWidth> attended{};
    std::copy(blocks.begin(), blocks.end(), attended.begin());
    // The lists' located blocks resolve from spilled history pages, which
    // the projection's stores into the step's own slots do not touch.
    if (step.spilled.active()) {
      if (blocks.empty() || !blocks.front().stride)
        throw std::logic_error("a decode step over spilled K/V attends without block lists");
      // The lanes' lists lie in one buffer, the lists of each lane's tiles at
      // the lane cap after the lanes before it (Qwen4ExpTarget::addDecodeQsa),
      // which the reads locate at once.
      ops::AttentionBlocks command = blocks.front();
      command.list = 0;
      const ops::AttentionBlocks located =
          step.spilled.verify(step.graph, layer, command, step.attention->sequences().front().tiles());
      for (size_t index = 0; index < blocks.size(); ++index) {
        attended[index] = located;
        attended[index].list = blocks[index].list;
      }
    }
    for (size_t index = 0; index < sequences.size(); ++index)
      histories[index] = {step.kvLayers[layer], pageTables[index], attended[index]};
  } else {
    for (size_t index = 0; index < step.sequences.size(); ++index) {
      const QwenTargetPrefillSequence &sequence = step.sequences[index];
      const uint32_t firstPage = sequence.chunk.committed_tokens / kv::kPageTokens;
      const PrefillKv history =
          step.spilled.prefill(step.graph, layer, sequence.lane, firstPage,
                               kv::chunkedPrefillRequiredPages(sequence.chunk) - firstPage,
                               {sequence.pageTable, step.kvLayers[layer]});
      histories[index] = {history.kv, history.table, blocks.empty() ? ops::AttentionBlocks{} : blocks[index]};
    }
  }
  // Only each sequence's last row is read (planOutputRows): the tile that
  // holds it attends as in the chunk (lastTiles), and its gated row is
  // written row `index`.
  const bool lastRows = !step.readRows.empty();
  const ops::AttentionPlan &plan = variant.windowTokens ? *step.windowedAttention : *step.attention;
  const ops::LinearPlan outputPlan =
      linear.plan(mixer.outputProjection, step.outputRows(),
                  output.residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None, nullptr, step.rows);
  const ops::PreparedInput hidden = ops::PagedAttention::addAttention(
      step.graph, lastRows ? ops::PagedAttention::lastTiles(plan) : plan, b.fullQueries, b.attentionPartials,
      b.attentionStatistics, sequences, std::span(histories).first(sequences.size()), rows,
      {b.fullPacked, b.attentionHidden, lastRows ? ops::AttentionOutput::Last : ops::AttentionOutput::Every,
       b.linearScratch, lastRows ? ops::LinearInput::Plain : outputPlan.input()},
      mixer.sinks);
  observe(step.graph, "attention_gated_hidden", b.attentionHidden, step.outputRows(), geometry_.attentionWidth);
  // The read rows' residual rows, gathered into the normalized rows the
  // input projection has read.
  metal::MetalBuffer residual = std::move(output.residual);
  if (lastRows) {
    for (uint32_t row = 0; row < step.readRows.size(); ++row)
      ops::RowCopy::add(step.graph, residual, {step.readRows[row], geometry_.hiddenSize, 0}, b.normalized,
                        {row, geometry_.hiddenSize, 0}, 1, geometry_.hiddenSize);
    residual = b.normalized;
  }
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = std::move(output.rows), .residual = std::move(residual),
              .scratch = b.linearScratch, .prepared = hidden},
             mixer.outputProjection, outputPlan);
}

// The norm writes the rows as the FFN's first plan takes them (swiGluInput):
// a decode step's and a short chunk's gate/up plan, a longer chunk's gate.
void QwenTarget::addPrefillDenseFfn(PrefillStep &step, uint32_t index, const ops::NormWeights &norm,
                                    ops::SwiGluProjections ffn, metal::MetalBuffer residual,
                                    metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t rows = step.outputRows();
  const ops::PrefillFfnBuffers buffers = b.ffn();
  // The split takes whole chunks; read rows alone run on the GPU.
  if (step.aneFfn && step.readRows.empty()) {
    ops::Normalization::addRms(step.graph, residual, norm, b.normalized, geometry_.hiddenSize, rows);
    step.aneFfn->add(step.graph, index, buffers, std::move(residual), std::move(output), rows);
    return;
  }
  const ops::PreparedInput normalized =
      ops::Normalization::addRms(step.graph, residual, norm, b.normalized, geometry_.hiddenSize, rows,
                                 b.linearScratch, linear.swiGluInput(ffn, rows, step.rows));
  linear.addSwiGlu(step.graph, ffn, buffers, std::move(residual), std::move(output), rows, step.rows, normalized);
}

// A decode step's lanes route their rows alone (ops::MoeBuffers::lanes).
void QwenTarget::addPrefillMoeRouting(PrefillStep &step, const ops::NormWeights &norm,
                                      const ops::BlockMoeWeights &ffn, metal::MetalBuffer residual,
                                      metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, norm, b.normalized, geometry_.hiddenSize, step.outputRows());
  observe(step.graph, "moe_norm", b.normalized, step.outputRows(), geometry_.hiddenSize);
  step.ffn = {.input = b.normalized, .residual = std::move(residual), .output = std::move(output), .scratch = b.moe};
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    step.ffn.lanes = d->step.lanes;
    step.ffn.laneRows = b.laneRows;
  }
  if (step.grouped) return;
  ops::MoE::addRouting(step.graph, step.ffn, ffn, *step.moe);
  observe(step.graph, "moe_route_weights", b.moe.routingWeights, step.outputRows(), step.moe->shape().routesPerToken(), 4);
}

void QwenTarget::addNormalizedHead(metal::CommandGraph &graph, const ops::NormWeights &finalNorm,
                                   metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                                   metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const {
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan plan = linear.plan(vocabularyProjection(), rows);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), finalNorm, finalHidden, geometry_.hiddenSize, rows, scratch, plan.input());
  observe(graph, "head_norm", finalHidden, rows, geometry_.hiddenSize);
  linear.add(graph,
             {.input = finalHidden, .output = logits, .scratch = scratch,
              .prepared = normalized},
             vocabularyProjection(), plan);
  observe(graph, "target_logits", logits, rows, geometry_.vocabularySize, 4);
}

void QwenTarget::addCapture(metal::CommandGraph &graph, uint32_t slot, metal::MetalBuffer output,
                            uint32_t sourceRow, metal::MetalBuffer captured, uint32_t destinationRow,
                            uint32_t rows) const {
  const uint32_t width = geometry_.hiddenSize, capturedWidth = geometry_.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("Qwen target capture slot past the captured hidden rows");
  ops::RowCopy::add(graph, std::move(output), {sourceRow, width, 0}, std::move(captured),
                    {destinationRow, capturedWidth, slot * width}, rows, width);
}

namespace {

// The rows a mixer's output projection adds to the layer's input rows into.
template <class Buffers> metal::MetalBuffer mixerOutput(const Buffers &b, const QwenGdnWeights &) {
  return b.gdnOutput;
}
template <class Buffers> metal::MetalBuffer mixerOutput(const Buffers &b, const QwenAttentionWeights &) {
  return b.attentionOutput;
}

// The Qwen3.5 architecture: Qwen3.8-27B (model/Qwen3_8.hpp, a dense FFN),
// Qwen3.6-35B-A3B (model/Qwen3_6Moe.hpp, a sparse MoE) and Flint
// (model/Flint.hpp: every layer attends, with its sinks, and its FFN is dense
// on the leading layers and a MoE block on the others). Layer l reads its
// input rows from buffers.hidden[l & 1]; its mixer and its FFN each read the
// RMS norm of the rows before them and add their output to them, into the
// other buffer. The layers the draft reads copy their output rows into the
// captured rows, and the head reads the final norm of the last layer's.
template <class Layout, class Layer> class Qwen3_5Target final : public QwenTarget {
public:
  Qwen3_5Target(const QwenTargetWeights<Layout, Layer> &weights, const QwenTargetGeometry &geometry,
                metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
      : QwenTarget(weights, geometry, backend, operators), weights_(weights) {
    requireLayers(std::span<const Layer>(weights.layers));
  }

  std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph, std::span<const QwenTargetPrefillChunk> chunks,
                                             const kv::PageStorage &kv, SpilledKvReads &spilled,
                                             QwenPrefillLayers layers, ops::AneFfn *aneFfn) override {
    std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, spilled, layers, aneFfn);
    return addPrefillLayers(std::span(steps), *this, layers);
  }

  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const override {
    addNormalizedHead(graph, weights_.finalNorm, std::move(hidden), std::move(finalHidden), std::move(logits), rows,
                      scratch);
  }

private:
  // The shared prefill's layer loop runs the family's (QwenTarget::addPrefillLayers).
  friend class QwenTarget;

  void addPrefillBegin(PrefillStep &) const {}
  [[nodiscard]] bool attends(uint32_t index) const {
    return std::holds_alternative<QwenAttentionWeights>(weights_.layers[index].mixer);
  }
  // A layer whose output work runs on read rows alone (planOutputRows) ends
  // them over its input's first rows, which its mixer has read, and
  // addPrefillExperts places them.
  void addPrefillLayer(PrefillStep &step, uint32_t index) const {
    const Layer &layer = weights_.layers[index];
    const QwenTargetPrefillBuffers &b = step.buffers;
    const metal::MetalBuffer input = b.hidden[index & 1];
    if (warmupTrace_) {
      warmupTrace_->layer = index;
      warmupTrace_->laneRows = step.decode ? b.laneRows : metal::MetalBuffer{};
      observe(step.graph, "layer_input", input, step.rows, geometry_.hiddenSize);
    }
    planOutputRows(step, index);
    const metal::MetalBuffer residual = std::visit(
        [&](const auto &mixer) {
          const ops::PreparedInput normalized = ops::Normalization::addRms(
              step.graph, input, layer.inputNorm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch,
              operators_.linear().plan(mixer.inputProjection, step.rows).input());
          observe(step.graph, "mixer_norm", b.normalized, step.rows, geometry_.hiddenSize);
          const BlockOutput block{mixerOutput(b, mixer), input};
          addPrefillMixer(step, mixer, normalized, block);
          return block.rows;
        },
        layer.mixer);
    observe(step.graph, "mixer_residual", residual, step.outputRows(), geometry_.hiddenSize);
    addPrefillFfn(step, index, layer, residual, step.readRows.empty() ? b.hidden[(index & 1) ^ 1] : input);
  }
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const { return qwenMoeBlock(weights_.layers[index]); }
  void addPrefillExperts(PrefillStep &step, uint32_t index) const {
    const QwenTargetPrefillBuffers &b = step.buffers;
    if (const ops::BlockMoeWeights *ffn = moe(index); ffn && !step.grouped)
      ops::MoE::addRouted(step.graph, step.ffn, *ffn, *step.moe);
    if (warmupTrace_ && step.moe) {
      const auto shape = step.moe->shape();
      const auto routes = shape.routesPerToken(), rows = step.outputRows();
      observe(step.graph, "moe_grouped_input", b.moe.groupedInput, rows, shape.hiddenSize, 2, routes, b.moe.routeRows);
      observe(step.graph, "moe_gate_up", b.moe.expertIntermediate, rows, shape.expertIntermediateSize, 2, routes, b.moe.routeRows);
      observe(step.graph, "moe_down", b.moe.expertOutput, rows, shape.hiddenSize, 2, routes, b.moe.routeRows);
      observe(step.graph, "moe_combine", step.ffn.output, rows, shape.hiddenSize);
    }
    addReadRows(step, b.hidden[index & 1], b.hidden[(index & 1) ^ 1]);
    // The draft reads a decode step's every row of the layers it captures.
    if (step.decode) {
      if (const auto slot = geometry_.captureSlot(index))
        addCapture(step.graph, *slot, b.hidden[(index & 1) ^ 1], 0, step.decode->capturedTargetHidden, 0, step.rows);
      return;
    }
    forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
      addCapture(step.graph, slot, b.hidden[(index & 1) ^ 1], source, b.captured, destination, rows);
    });
  }
  // A decode step's head reads every row's final norm.
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const {
    const metal::MetalBuffer hidden = step.buffers.hidden[geometry_.layers & 1];
    if (const QwenTargetDecodeBuffers *d = step.decode)
      addNormalizedHead(step.graph, weights_.finalNorm, hidden, d->finalHidden, d->logits, step.rows,
                        step.buffers.linearScratch);
    return hidden;
  }

  // The FFN of layer `index` (QwenTarget::addPrefillDenseFfn and its kin): a
  // dense FFN, or a MoE's routing, whose routed work addPrefillExperts adds.
  void addPrefillFfn(PrefillStep &step, uint32_t index, const Layer &layer, metal::MetalBuffer residual,
                     metal::MetalBuffer output) const {
    if (const std::optional<ops::SwiGluProjections> dense = denseFfn(layer))
      addPrefillDenseFfn(step, index, layer.postAttentionNorm, *dense, std::move(residual), std::move(output));
    else
      addPrefillMoeRouting(step, layer.postAttentionNorm, *qwenMoeBlock(layer), std::move(residual),
                           std::move(output));
  }
  const QwenTargetWeights<Layout, Layer> &weights_;
};

} // namespace

template <class Layout, class Layer>
std::unique_ptr<QwenTarget> makeQwenTarget(const QwenTargetWeights<Layout, Layer> &weights,
                                           const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                                           const ops::ExecutionPlans &operators, uint32_t) {
  return std::make_unique<Qwen3_5Target<Layout, Layer>>(weights, geometry, backend, operators);
}

template std::unique_ptr<QwenTarget> makeQwenTarget(const Qwen3_8Weights &, const QwenTargetGeometry &,
                                                    metal::MetalBackend &, const ops::ExecutionPlans &, uint32_t);
template std::unique_ptr<QwenTarget> makeQwenTarget(const Qwen3_6MoeWeights &, const QwenTargetGeometry &,
                                                    metal::MetalBackend &, const ops::ExecutionPlans &, uint32_t);
template std::unique_ptr<QwenTarget> makeQwenTarget(const FlintWeights &, const QwenTargetGeometry &,
                                                    metal::MetalBackend &, const ops::ExecutionPlans &, uint32_t);

} // namespace splash::model
