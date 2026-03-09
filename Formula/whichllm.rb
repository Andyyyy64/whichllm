class Whichllm < Formula
  include Language::Python::Virtualenv

  desc "Find the best local LLM that actually runs on your hardware"
  homepage "https://github.com/Andyyyy64/whichllm"
  url "https://pypi.io/packages/source/w/whichllm/whichllm-0.3.0.tar.gz"
  sha256 "PLACEHOLDER"
  license "MIT"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "whichllm", shell_output("#{bin}/whichllm --version")
  end
end
