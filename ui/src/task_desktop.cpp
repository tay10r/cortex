#include "task.h"

#include <map>
#include <string>
#include <thread>

#include <boost/beast.hpp>

namespace cortex {

namespace {

struct response final
{
  std::string body;

  std::map<std::string, std::string> headers;
};

[[nodiscard]] auto
make_image_response() -> response
{
  const auto w{ 3280 };
  const auto h{ 2464 };
  const auto num_pixels{ w * h };

  response resp;
  resp.headers["x-image-width"] = std::to_string(w);
  resp.headers["x-image-height"] = std::to_string(h);
  resp.body.resize(w * h * 2);

  auto* data = reinterpret_cast<uint16_t*>(resp.body.data());

  for (auto i = 0; i < num_pixels; i++) {

    const auto x{ i % w };
    const auto y{ i / w };

    const auto u{ (static_cast<float>(x) + 0.5F) / static_cast<float>(w) };
    const auto v{ (static_cast<float>(y) + 0.5F) / static_cast<float>(h) };

    const auto r{ u };
    const auto g{ v };
    const auto b{ 1.0F };

    const auto j = (x % 2) + (y % 2) * 2;

    float color{ 0.0F };

    switch (j) {
      case 0:
        // green
        color = g;
        break;
      case 1:
        // blue
        color = b;
        break;
      case 2:
        // red
        color = r;
        break;
      case 3:
        // green 2
        color = g;
        break;
    }

    data[i] = static_cast<int>(color * 1023.0F);
  }

  return resp;
}

[[nodiscard]] auto
make_sangaboard_response() -> response
{
  response resp;
  resp.body = "done.";
  return resp;
}

[[nodiscard]] auto
make_images_response() -> response
{
  response resp;
  resp.body = R"(
{
  "28f8d13e-ee0f-4d9a-99e8-6618eee93210": {
    "width": 100,
    "height": 100,
    "creation_time": 1746716690
  },
  "c9abf362-7fa9-429e-9006-4e94b7e89c78": {
    "width": 120,
    "height": 80,
    "creation_time": 1746716790
  },
  "4df69c2a-9f1f-44b1-9356-3a6fc7c41f65": {
    "width": 90,
    "height": 90,
    "creation_time": 1746716890
  },
  "f3d0e438-97bb-4b4d-b4e4-26979b43ddc4": {
    "width": 150,
    "height": 150,
    "creation_time": 1746716990
  },
  "69a37fa0-42e1-4c16-84e7-38eaec4bc690": {
    "width": 80,
    "height": 120,
    "creation_time": 1746717090
  },
  "b845e5aa-41bc-4d33-8120-f48a7b73026a": {
    "width": 110,
    "height": 95,
    "creation_time": 1746717190
  },
  "e7dffb2e-17b7-4636-bad1-b1aee9bc645e": {
    "width": 200,
    "height": 200,
    "creation_time": 1746717290
  }
}
  )";
  return resp;
}

[[nodiscard]] auto
make_config_get_response() -> response
{
  response resp;
  resp.body = std::string("\x00\x00\x00\x00\x00\x00\x00\x00", static_cast<size_t>(8));
  return resp;
}

class task_impl final : public task
{
  std::string url_;

  float progress_{ 0.0F };

  response response_;

  bool done_{ false };

  bool failed_{ false };

public:
  explicit task_impl(std::string url)
    : url_(std::move(url))
  {
  }

  auto done() const -> bool override { return done_; }

  auto failed() const -> bool override { return failed_; }

  auto data() const -> const void* override { return response_.body.data(); }

  auto size() const -> size_t override { return response_.body.size(); }

  void poll() override
  {
    // simulate progress
    if (progress_ < 1.0F) {
      progress_ += 0.02F;
      if (progress_ >= 1.0F) {
        complete_response();
      }
    }
  }

  [[nodiscard]] auto progress() const -> float override { return progress_; }

  [[nodiscard]] auto get_header_value(const char* key) const -> const char* override
  {
    auto it = response_.headers.find(key);
    if (it == response_.headers.end()) {
      return nullptr;
    }
    return it->second.c_str();
  }

protected:
  void complete_response()
  {
    if (url_.find("snapshot") != std::string::npos) {
      response_ = make_image_response();
    } else if (url_.find("command") != std::string::npos) {
      response_ = make_sangaboard_response();
    } else if (url_.find("images.json") != std::string::npos) {
      response_ = make_images_response();
    } else if (url_.find("config") != std::string::npos) {
      response_ = make_config_get_response();
    } else {
      failed_ = true;
    }

    done_ = true;
  }
};

class threaded_task : public task
{
public:
};

} // namespace

auto
task::http_post(const std::string& url, std::string) -> std::unique_ptr<task>
{
  return std::make_unique<task_impl>(url);
}

auto
task::http_get(const std::string& url) -> std::unique_ptr<task>
{
  return std::make_unique<task_impl>(url);
}

auto
task::http_delete(const std::string& url) -> std::unique_ptr<task>
{
  return std::make_unique<task_impl>(url);
}

auto
task::http_put(const std::string& url, std::string) -> std::unique_ptr<task>
{
  return std::make_unique<task_impl>(url);
}

} // namespace cortex
