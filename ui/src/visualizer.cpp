#include "visualizer.h"

#include "image.h"

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

struct uniform_map final
{
  GLint bayer_data_location{};

  GLint texture_size_location{};

  GLint texel_size_location{};

  GLint color_balance_location{};

  GLint gain_location{};

  GLint gamma_location{};
};

class shader final
{
  GLuint id_{};

  GLint position_location_{};

  uniform_map uniforms_;

public:
  explicit shader(const std::string& frag_src)
    : id_(uikit::compile_shader(get_shader("/shaders/debayer.vert").c_str(), frag_src.c_str(), {}))
  {
    position_location_ = glGetAttribLocation(id_, "a_position");

    auto& u = uniforms_;
    u.bayer_data_location = glGetUniformLocation(id_, "bayer_data");
    u.texture_size_location = glGetUniformLocation(id_, "texture_size");
    u.texel_size_location = glGetUniformLocation(id_, "texel_size");
    u.color_balance_location = glGetUniformLocation(id_, "color_balance");
    u.gain_location = glGetUniformLocation(id_, "gain");
    u.gamma_location = glGetUniformLocation(id_, "gamma");
  }

  void use() { glUseProgram(id_); }

  [[nodiscard]] auto get_position_location() const -> GLint { return position_location_; }

  [[nodiscard]] auto get_uniforms() const -> const uniform_map* { return &uniforms_; }
};

class visualizer_impl final : public visualizer
{
  void* parent_{};

  plot_callback plot_cb_{};

  GLuint vertex_buffer_{};

  GLuint bayer_texture_{};

  GLuint color_attachment_{};

  GLuint framebuffer_{};

  TextEditor editor_;

  std::unique_ptr<shader> shader_;

  std::string compile_error_;

  float balance_[3]{ 0.7F, 1.0F, 2.0F };

  float gain_{ 1.0F };

  float gamma_{ 2.2F };

  int width_{};

  int height_{};

  image image_;

public:
  visualizer_impl(void* parent, plot_callback plot_cb)
    : parent_(parent)
    , plot_cb_(plot_cb)
  {
  }

  void setup() override
  {
    glGenBuffers(1, &vertex_buffer_);

    glGenTextures(1, &bayer_texture_);
    glBindTexture(GL_TEXTURE_2D, bayer_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#ifdef __EMSCRIPTEN__
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#else
    /* Note: This border behavior is important for accurate interpolation at the edges */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
#endif

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

      if (ImGui::BeginTabItem("Shader")) {
        loop_settings();
        ImGui::EndTabItem();
      }

      ImGui::EndTabBar();
    }
  }

  void update(const void* bayer_data, const int w, const int h) override
  {
    image_ = image(static_cast<const uint16_t*>(bayer_data), w, h);

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

    const auto* u = shader_->get_uniforms();
    glUniform2f(u->texture_size_location, static_cast<float>(w), static_cast<float>(h));
    glUniform2f(u->texel_size_location, 1.0F / static_cast<float>(w), 1.0F / static_cast<float>(h));
    glUniform3f(u->color_balance_location, balance_[0], balance_[1], balance_[2]);
    glUniform1f(u->gain_location, gain_);
    glUniform1f(u->gamma_location, gamma_);
    glUniform1i(u->bayer_data_location, 0);

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
    if (ImGui::InputFloat3("Color Balance", balance_)) {
      render_frame();
    }

    if (ImGui::DragFloat("Digital Gain", &gain_, /*v_speed=*/0.02F, /*v_min=*/1.0F, /*v_max=*/10.0F)) {
      render_frame();
    }

    if (ImGui::DragFloat("Gamma", &gamma_, /*v_speed=*/0.01F, /*v_min=*/0.1F, /*v_max=*/10.0F)) {
      render_frame();
    }

    if (!ImPlot::BeginPlot("##Viewport", ImVec2(-1, -1), ImPlotFlags_Equal | ImPlotFlags_CanvasOnly)) {
      return;
    }

    ImPlot::PlotImage(
      "##Image", reinterpret_cast<ImTextureID>(color_attachment_), ImPlotPoint(0, 0), ImPlotPoint(width_, height_));

    if (plot_cb_) {
      plot_cb_(parent_, image_);
    }

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
visualizer::create(void* parent, plot_callback plot_cb) -> std::unique_ptr<visualizer>
{
  return std::make_unique<visualizer_impl>(parent, plot_cb);
}

} // namespace cortex
