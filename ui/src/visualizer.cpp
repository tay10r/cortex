#include "visualizer.h"

#include <uikit/shader_compiler.hpp>

#include <GLES2/gl2.h>

#include <imgui.h>

#include <implot.h>

#include <TextEditor.h>

#include <cmrc/cmrc.hpp>

CMRC_DECLARE(shaders);

namespace cortex {

namespace {

[[nodiscard]] auto
get_shader(const std::string& path) -> std::string
{
  const auto fs = cmrc::shaders::get_filesystem();
  const auto file = fs.open(path);
  return std::string(file.begin(), file.size());
}

class shader final
{
  GLuint id_{};

  GLint position_location_{};

  GLint bayer_data_location_{};

  GLint texture_size_location_{};

  GLint texel_size_location_{};

public:
  explicit shader(const std::string& frag_src)
    : id_(uikit::compile_shader(get_shader("/shaders/debayer.vert").c_str(), frag_src.c_str(), {}))
  {
    position_location_ = glGetAttribLocation(id_, "a_position");
    bayer_data_location_ = glGetUniformLocation(id_, "bayer_data");
    texture_size_location_ = glGetUniformLocation(id_, "texture_size");
    texel_size_location_ = glGetUniformLocation(id_, "texel_size");
  }

  void use() { glUseProgram(id_); }

  [[nodiscard]] auto get_position_location() const -> GLint { return position_location_; }

  [[nodiscard]] auto get_bayer_data_location() const -> GLint { return bayer_data_location_; }

  [[nodiscard]] auto get_texture_size_location() const -> GLint { return texture_size_location_; }

  [[nodiscard]] auto get_texel_size_location() const -> GLint { return texel_size_location_; }
};

class visualizer_impl final : public visualizer
{
  GLuint vertex_buffer_{};

  GLuint bayer_texture_{};

  GLuint color_attachment_{};

  GLuint framebuffer_{};

  TextEditor editor_;

  std::unique_ptr<shader> shader_;

  std::string compile_error_;

  int width_{};

  int height_{};

public:
  void setup() override
  {
    glGenBuffers(1, &vertex_buffer_);

    glGenTextures(1, &bayer_texture_);
    glBindTexture(GL_TEXTURE_2D, bayer_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    /* Note: This border behavior is important for accurate interpolation at the edges */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    glGenTextures(1, &color_attachment_);
    glBindTexture(GL_TEXTURE_2D, color_attachment_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &framebuffer_);

    editor_.SetLanguageDefinition(TextEditor::LanguageDefinition::GLSL());

    editor_.SetText(get_shader("/shaders/debayer.frag"));

    compile_shader();

    setup_vertex_buffer();

    setup_image_buffers(3280, 2464);
  }

  void teardown() override
  {
    glDeleteBuffers(1, &vertex_buffer_);

    glDeleteTextures(1, &bayer_texture_);

    glDeleteTextures(1, &color_attachment_);

    glDeleteFramebuffers(1, &framebuffer_);
  }

  void loop() override
  {
    if (ImGui::BeginTabBar("Tabs")) {

      if (ImGui::BeginTabItem("Viewport")) {
        loop_viewport();
        ImGui::EndTabItem();
      }

      if (ImGui::BeginTabItem("Settings")) {
        loop_settings();
        ImGui::EndTabItem();
      }

      ImGui::EndTabBar();
    }
  }

  void update(const void* bayer_data, const int w, const int h) override
  {
    setup_image_buffers(w, h);

    glActiveTexture(GL_TEXTURE0);

    glBindTexture(GL_TEXTURE_2D, bayer_texture_);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w / 2, h, GL_RGBA, GL_UNSIGNED_BYTE, bayer_data);

    render_frame();
  }

protected:
  void render_frame()
  {
    if (!shader_) {
      return;
    }

    const auto w{ width_ };
    const auto h{ height_ };

    shader_->use();

    glUniform2f(shader_->get_texture_size_location(), static_cast<float>(w), static_cast<float>(h));
    glUniform2f(shader_->get_texel_size_location(), 1.0F / static_cast<float>(w), 1.0F / static_cast<float>(h));

    // point texture uniform to GL_TEXTURE0 (we probably don't need to, but should for correctness)
    glUniform1i(shader_->get_bayer_data_location(), 0);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);

    glViewport(0, 0, width_, height_);

    const auto position_location = shader_->get_position_location();

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);

    glEnableVertexAttribArray(position_location);

    glVertexAttribPointer(position_location, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, nullptr);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  void loop_viewport()
  {
    if (!ImPlot::BeginPlot("##Viewport", ImVec2(-1, -1), ImPlotFlags_Equal | ImPlotFlags_CanvasOnly)) {
      return;
    }

    ImPlot::PlotImage(
      "##Image", reinterpret_cast<ImTextureID>(color_attachment_), ImPlotPoint(0, 0), ImPlotPoint(width_, height_));

    ImPlot::EndPlot();
  }

  void loop_settings()
  {
    if (ImGui::Button("Compile")) {
      compile_shader();
    }

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));

    ImGui::TextUnformatted(compile_error_.c_str());

    ImGui::PopStyleColor();

    editor_.Render("Fragment Shader");
  }

  void compile_shader()
  {
    compile_error_.clear();

    try {
      shader_ = std::make_unique<shader>(editor_.GetText().c_str());
    } catch (const std::exception& e) {
      compile_error_ = e.what();
    }
  }

  void setup_vertex_buffer()
  {
    const float vertices[] = {
      // clang-format off
      0, 0,
      1, 0,
      1, 1,
      0, 0,
      1, 1,
      0, 1
      // clang-format on
    };

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);

    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  }

  void setup_image_buffers(const int w, const int h)
  {
    if ((w == width_) && (h == height_)) {
      return;
    }

    glBindTexture(GL_TEXTURE_2D, bayer_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w / 2, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, color_attachment_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_attachment_, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    width_ = w;
    height_ = h;
  }
};

} // namespace

auto
visualizer::create() -> std::unique_ptr<visualizer>
{
  return std::make_unique<visualizer_impl>();
}

} // namespace cortex
