#include "transfer.h"

#include <emscripten/fetch.h>

#include <map>
#include <string>

namespace cortex {

namespace {

class transfer_impl final : public transfer
{
  emscripten_fetch_attr_t attr_{};

  emscripten_fetch_t* fetch_{};

  bool done_{ false };

  bool failed_{ false };

  std::map<std::string, std::string> headers_;

  std::string body_;

public:
  explicit transfer_impl(const char* method, const std::string& url, std::string body)
    : body_(std::move(body))
  {
    strcpy(attr_.requestMethod, method);

    attr_.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr_.onsuccess = on_success;
    attr_.onerror = on_error;

    if (!body_.empty()) {
      attr_.requestData = body_.data();
      attr_.requestDataSize = body_.size();
    }

    fetch_ = emscripten_fetch(&attr_, url.c_str());
    fetch_->userData = this;
  }

  ~transfer_impl()
  {
    if (fetch_) {
      emscripten_fetch_close(fetch_);
    }
  }

  [[nodiscard]] auto data() const -> const void* override { return fetch_->data; }

  [[nodiscard]] auto size() const -> size_t override { return fetch_->numBytes; }

  [[nodiscard]] auto done() const -> bool override { return done_; }

  [[nodiscard]] auto failed() const -> bool override { return failed_; }

  [[nodiscard]] auto progress() const -> float override
  {
    if (fetch_->totalBytes > 0) {
      return static_cast<float>(fetch_->dataOffset) / static_cast<float>(fetch_->totalBytes);
    } else {
      return 0.0F;
    }
  }

  [[nodiscard]] auto get_header_value(const char* key) const -> const char* override
  {
    auto it = headers_.find(key);
    if (it == headers_.end()) {
      return nullptr;
    }
    return it->second.c_str();
  }

  void poll() override {}

protected:
  static auto get_self(emscripten_fetch_t* fetch) -> transfer_impl*
  {
    return static_cast<transfer_impl*>(fetch->userData);
  }

  static void on_success(emscripten_fetch_t* fetch)
  {
    auto* self = get_self(fetch);
    self->done_ = true;
    self->read_headers();
  }

  static void on_error(emscripten_fetch_t* fetch)
  {
    auto* self = get_self(fetch);
    self->done_ = true;
    self->failed_ = true;
  }

  void read_headers()
  {
    const auto len{ emscripten_fetch_get_response_headers_length(fetch_) };
    std::string data;
    data.resize(len);
    emscripten_fetch_get_response_headers(fetch_, data.data(), data.size());
    auto* headers = emscripten_fetch_unpack_response_headers(data.data());
    for (size_t i = 0; headers[i] != nullptr; i += 2) {
      const auto* key{ headers[i] };
      const auto* val{ headers[i + 1] };
      headers_.emplace(key, val);
    }
    emscripten_fetch_free_unpacked_response_headers(headers);
  }
};

} // namespace

auto
transfer::post(const std::string& url, std::string body) -> std::unique_ptr<transfer>
{
  return std::make_unique<transfer_impl>("POST", url, std::move(body));
}

auto
transfer::get(const std::string& url) -> std::unique_ptr<transfer>
{
  return std::make_unique<transfer_impl>("GET", url, "");
}

auto
transfer::delete_(const std::string& url) -> std::unique_ptr<transfer>
{
  return std::make_unique<transfer_impl>("DELETE", url, "");
}

auto
transfer::put(const std::string& url) -> std::unique_ptr<transfer>
{
  return std::make_unique<transfer_impl>("PUT", url, "");
}

} // namespace cortex
