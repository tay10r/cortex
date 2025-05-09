#include "image_index.h"

#include "transfer.h"

#include <json.hpp>

#include <algorithm>
#include <sstream>

namespace cortex {

namespace {

class image_index_impl final : public image_index
{
  std::vector<image_info> image_info_;

  std::unique_ptr<transfer> refresh_transfer_;

public:
  void loop() override
  {
    if (!refresh_transfer_) {
      return;
    }

    refresh_transfer_->poll();

    if (refresh_transfer_->done()) {

      if (!refresh_transfer_->failed()) {
        handle_refresh(static_cast<const char*>(refresh_transfer_->data()), refresh_transfer_->size());
      }

      refresh_transfer_.reset();
    }
  }

  [[nodiscard]] auto refreshing() const -> bool override { return !!refresh_transfer_; }

  void refresh() override
  {
    if (refresh_transfer_) {
      return;
    }

    refresh_transfer_ = transfer::get("/images.json");
  }

  auto get_image_info() -> std::vector<image_info>& override { return image_info_; }

  auto get_image_info() const -> const std::vector<image_info>& override { return image_info_; }

protected:
  [[nodiscard]] static auto timestamp_to_string(uint64_t unix_timestamp) -> std::string
  {
    static_assert(sizeof(time_t) == sizeof(uint64_t), "size of time_t is invalid");
    const auto t = static_cast<time_t>(unix_timestamp);
    std::tm* tm_ptr = std::localtime(&t); // Convert to local time
    std::ostringstream oss;
    oss << std::put_time(tm_ptr, "%Y-%m-%d %H:%M:%S");
    return oss.str();
  }

  void handle_refresh(const char* json, size_t size)
  {
    const auto root = nlohmann::json::parse(json, json + size);

    std::vector<image_info> next;

    for (const auto& entry : root.items()) {

      const auto& val = entry.value();

      image_info info;
      info.id = entry.key();
      info.width = val.at("width").get<size_t>();
      info.height = val.at("height").get<size_t>();
      info.creation_time = val.at("creation_time").get<uint64_t>();
      info.label = timestamp_to_string(info.creation_time) + "##" + info.id;

      next.emplace_back(std::move(info));
    }

    std::sort(next.begin(), next.end(), [](const image_info& a, const image_info& b) -> bool {
      return a.creation_time > b.creation_time;
    });

    image_info_ = std::move(next);
  }
};

} // namespace

auto
image_index::create() -> std::unique_ptr<image_index>
{
  return std::make_unique<image_index_impl>();
}

} // namespace cortex
