class Ytalk < Formula
  include Language::Python::Virtualenv

  desc "Download YouTube videos, transcribe with Whisper, and chat with Ollama"
  homepage "https://github.com/<user>/ytalk"
  url "https://github.com/<user>/ytalk/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "<sha256>"
  license "MIT"

  depends_on "python@3.12"
  depends_on "ffmpeg"

  # Resource stanzas generated via: poet ytalk
  # Run: pip install homebrew-pypi-poet && poet ytalk
  # Then paste the output here (50+ entries due to torch/whisper dependency chain)

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      ytalk requires Ollama for summarization and chat.
      Install from https://ollama.ai, then: ollama pull gemma3:4b
    EOS
  end

  test do
    assert_match "usage", shell_output("#{bin}/ytalk --help", 2)
  end
end
