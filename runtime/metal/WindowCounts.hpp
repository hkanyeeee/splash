#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::metal {
// One fresh private backing file per measured arm. Counters describe encoded
// dispatches. Only successful, completed requests can qualify for comparison.
class WindowCounts {
  static constexpr size_t bytes = 4 * sizeof(uint64_t);
  uint64_t *words = nullptr;
public:
  WindowCounts() {
    const char *name = std::getenv("SPLASH_231_COUNTER_FILE");
    if (!name) return;
    int fd = open(name, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    struct stat st{};
    if (fd < 0 || fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_nlink != 1 ||
        st.st_uid != geteuid() || st.st_size != bytes || (st.st_mode & 0777) != 0600) {
      if (fd >= 0) close(fd);
      throw std::runtime_error("invalid window counter file");
    }
    auto *mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) throw std::runtime_error("cannot map window counters");
    words = static_cast<uint64_t *>(mapped);
  }
  ~WindowCounts() { if (words) munmap(words, bytes); }
  WindowCounts(const WindowCounts &) = delete;
  void encoded(std::string_view name) const noexcept {
    if (!words) return;
    unsigned index;
    if (name.starts_with("m4_borrow_pair_")) index = 0;
    else if (name == "attention_bf16_split_flags") index = 1;
    else if (name == "attention_bf16_split") index = 2;
    else if (name == "draft_verify_rows") index = 3;
    else return;
    std::atomic_ref<uint64_t>(words[index]).fetch_add(1, std::memory_order_relaxed);
  }
};
}
