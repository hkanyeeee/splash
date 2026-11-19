#include "metal/CommandGraph.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

using namespace splash::metal;

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    MetalBackend backend(argv[1]);
    const auto word = [&](const char *name) {
      return backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared, name);
    };
    auto source = word("host-published word"), observed = word("GPU read after host"), result = word("GPU read after break");
    const auto value = [](const MetalBuffer &b) -> uint32_t & { return *static_cast<uint32_t *>(b.contents()); };
    value(source) = 7;
    value(observed) = value(result) = 0;
    // The load's consumer ends at an Apple7/8 prefill layer break. The GPU
    // must see the new value even if the host load finishes after encoding.
    CommandGraph graph;
    graph.host([source, &value] {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      value(source) = 73;
    }, {}, HostStepGpu::Idle, {}, {}, {source});
    graph.add("test_copy_u32", {source, observed}, uint32_t{1}, {1, 1, 1}, {1, 1, 1});
    graph.split();
    graph.add("test_copy_u32", {observed, result}, uint32_t{1}, {1, 1, 1}, {1, 1, 1});
    backend.preparePipelines(graph.dispatches());
    static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    std::printf("host=%u before_break=%u after_break=%u (expected 73 throughout)\n",
                value(source), value(observed), value(result));
    return value(source) == 73 && value(observed) == 73 && value(result) == 73 ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
}
