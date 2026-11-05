// Real cache policy, fake repository model. This measures no inference speed.
#define main original_engine_test_main
#include "tests/engine/engine_test.cpp"
#undef main

void probe(bool enabled, bool pressure, unsigned backing) {
  setenv("SPLASH_LOCAL_TAIL_CHECKPOINT", "message", 1);
  setenv("SPLASH_LOCAL_KEEP_MESSAGE_OFFLOAD", enabled ? "1" : "0", 1);
  test::TestKvStorage storage(8192, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  bool full = false;
  auto config = test::engineConfig();
  config.vocabularySize = 248320;
  config.maxContext = 262144;
  config.growthPaused = [&] { return full; };
  engine::Engine engine(config, cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(196320, 7);
  prompt[196224] = 248045;
  prompt[196225] = 846;
  prompt[196226] = 198;
  prompt[196312] = 248045;
  prompt[196313] = 74455;
  prompt[196314] = 198;
  auto fork = prompt;
  std::fill(fork.begin() + 196227, fork.begin() + 196310, 19);
  auto next = prompt;
  std::fill(next.begin() + 196227, next.begin() + 196310, 29);
  double now = 1;
  engine.submit(request(1, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "producer did not finish");
  require(cache.probe(fork, {}).cachedTokens() == 196224, "message checkpoint absent");
  uint64_t block = 0;
  {
    auto held = cache.lookup(fork, {});
    require(held.state.has_value(), "checkpoint lease absent");
    block = held.state->kvBlock();
  }
  bool wrote = false;
  auto control = std::make_shared<OffloadControl>();
  control->ready = true;
  if (backing == 1) {
    cache.invalidateState(block);
    cache.publishCompositeState(block, std::make_shared<OffloadState>(control, executor.evictedStateBytes), true);
  } else if (backing == 2) {
    cache.invalidateState(block);
    control->ready = false;
    require(cache.publishStateToDisk(block, [control](std::function<void()>) {
              return std::make_unique<OffloadTicket>(control);
            }, true), "fixture disk publication refused");
    cache.publishCompositeState(block, std::make_shared<State>(executor.evictedStateBytes), true);
    require(!cache.recycleMessageCheckpoint(cache.checkpointState(block), wrote) && !wrote,
            "pending disk copy was treated as durable");
    control->ready = true;
    require(cache.pollTransfers(), "fixture disk copy did not finish");
    require(cache.snapshot().stateCache.diskBytes == 64, "fixture disk copy absent");
  }
  const auto checkpoint = cache.checkpointState(block);
  require(!cache.recycleMessageCheckpoint({}, wrote) && !wrote, "empty handle changed cache");
  require(!cache.recycleMessageCheckpoint({block, checkpoint.publication + 1}, wrote) && !wrote,
          "stale handle changed cache");
  {
    auto held = cache.lookup(fork, {});
    require(held.state.has_value(), "pin fixture absent");
    require(!cache.recycleMessageCheckpoint(checkpoint, wrote) && !wrote,
            "pinned RAM checkpoint was recycled");
  }
  if (backing == 0)
    require(!cache.recycleMessageCheckpoint(checkpoint, wrote) && !wrote,
            "unwritable checkpoint was recycled");
  const bool retained = !pressure || (enabled && backing != 0);
  const auto before = engine.snapshot();
  full = pressure;
  *executor.evictedStateBytes = 0;
  executor.reservationFailure = metal::AllocationFailure::HostPressure;
  executor.reservationRoom = [&] {
    if (*executor.evictedStateBytes >= 64) {
      *executor.evictedStateBytes -= 64;
      return true;
    }
    return !full;
  };
  const uint32_t firstRows = executor.prefillRows;
  engine.submit(request(2, fork));
  tickUntil(engine, now, [&] { return idle(engine); }, "pressure fork did not finish");
  require(events.starts.back() == 196224 && events.failedCount == 0,
          "fork restored wrong state or failed");
  const auto after = engine.snapshot();
  const uint32_t resume = cache.probe(next, {}).cachedTokens();
  require(resume == (retained ? 196224U : 192512U), "checkpoint retention differs from specification");
  require(after.replayStatePublicationFailures == before.replayStatePublicationFailures,
          "replay publication failed");
  const bool newWrite = enabled && pressure && backing == 1;
  const uint64_t diskBytes = ((backing == 2 && retained) || newWrite) ? 64U : 0U;
  require(after.resources.stateCache.diskBytes == diskBytes, "disk lifetime differs");
  require(after.resources.stateCache.offloads - before.resources.stateCache.offloads == unsigned(newWrite),
          "new write count differs");
  require(after.resources.stateCache.offloadFailures == before.resources.stateCache.offloadFailures,
          "state write failed");
  const uint32_t forkRows = executor.prefillRows - firstRows;
  full = false;
  executor.restoreControl->ready = true;
  static_cast<void>(cache.pollTransfers());
  const uint32_t nextRows = executor.prefillRows;
  engine.submit(request(3, next));
  tickUntil(engine, now, [&] { return idle(engine); }, "next fork did not finish");
  require(events.starts.back() == resume && events.failedCount == 0,
          "saved checkpoint did not restore");
  const uint32_t computed = executor.prefillRows - nextRows;
  require(computed == (retained ? 96U : 3808U), "recomputed rows differ");
  std::cout << "{\"enabled\":" << enabled << ",\"pressure\":" << pressure
            << ",\"backing_kind\":" << backing << ",\"first_fork_rows\":" << forkRows
            << ",\"next_resume\":" << resume << ",\"next_computed_rows\":" << computed
            << ",\"capture_cuts\":" << after.captureCuts - before.captureCuts
            << ",\"checkpoint_retirements\":"
            << after.resources.stateCache.checkpointRetirements - before.resources.stateCache.checkpointRetirements
            << ",\"disk_bytes_after_fork\":" << after.resources.stateCache.diskBytes
            << ",\"new_writes\":" << unsigned(newWrite) << "}\n";
}

int main() {
  try {
    for (bool enabled : {false, true})
      for (bool pressure : {false, true})
        for (unsigned backing : {0U, 1U, 2U})
          probe(enabled, pressure, backing);
    unsetenv("SPLASH_LOCAL_TAIL_CHECKPOINT");
    setenv("SPLASH_LOCAL_KEEP_MESSAGE_OFFLOAD", "1", 1);
    require(original_engine_test_main() == 0, "full existing Engine suite failed");
    std::cout << "KEEP_MESSAGE_OFFLOAD_CPU_COMPLETE\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
