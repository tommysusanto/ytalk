class Ytalk < Formula
  include Language::Python::Virtualenv

  desc "Download YouTube videos, transcribe with Whisper, and chat with Ollama"
  homepage "https://github.com/tommysusanto/ytalk"
  url "https://github.com/tommysusanto/ytalk/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "bf1c6ba24b69ceec12d4b0e38e2e7422e4a6554a15290c295c54001414c37fd1"
  license "MIT"

  depends_on "ffmpeg"
  depends_on :macos
  depends_on "ollama"
  depends_on "python@3.12"

  on_arm do
    resource "llvmlite" do
      url "https://files.pythonhosted.org/packages/a2/9c/24139d3712d2d352e300c39c0e00d167472c08b3bd350c3c33d72c88ff8d/llvmlite-0.43.0-cp312-cp312-macosx_11_0_arm64.whl"
      sha256 "35d80d61d0cda2d767f72de99450766250560399edc309da16937b93d3b676e7"
    end

    resource "markupsafe" do
      url "https://files.pythonhosted.org/packages/9a/81/7e4e08678a1f98521201c3079f77db69fb552acd56067661f8c2f534a718/markupsafe-3.0.3-cp312-cp312-macosx_11_0_arm64.whl"
      sha256 "1872df69a4de6aead3491198eaf13810b565bdbeec3ae2dc8780f14458ec73ce"
    end

    resource "numba" do
      url "https://files.pythonhosted.org/packages/65/42/39559664b2e7c15689a638c2a38b3b74c6e69a04e2b3019b9f7742479188/numba-0.60.0-cp312-cp312-macosx_11_0_arm64.whl"
      sha256 "38d6ea4c1f56417076ecf8fc327c831ae793282e0ff51080c5094cb726507b1c"
    end

    resource "numpy" do
      url "https://files.pythonhosted.org/packages/75/5b/ca6c8bd14007e5ca171c7c03102d17b4f4e0ceb53957e8c44343a9546dcc/numpy-1.26.4-cp312-cp312-macosx_11_0_arm64.whl"
      sha256 "03a8c78d01d9781b28a6989f6fa1bb2c4f2d51201cf99d3dd875df6fbd96b23b"
    end

    resource "regex" do
      url "https://files.pythonhosted.org/packages/54/4b/ee27938d1b2c443e89a9a10e00d2d19aa5ee300cd3d61140644e93bb083e/regex-2026.5.9-cp312-cp312-macosx_11_0_arm64.whl"
      sha256 "f7a7c26137296beba7784de6eba69c6a93a63ccebc385e4962fe67e267a91225"
    end

    resource "tiktoken" do
      url "https://files.pythonhosted.org/packages/36/18/d4ac9d20956cdebca04841316660ed584c2fecdc2b81722a28bc7ad3b1e4/tiktoken-0.13.0-cp312-cp312-macosx_11_0_arm64.whl"
      sha256 "4d9980f11429ed2d737c463bb1fb78cf330caa026adf002f714aced7849a687b"
    end

    resource "torch" do
      url "https://files.pythonhosted.org/packages/4a/0e/e4e033371a7cba9da0db5ccb507a9174e41b9c29189a932d01f2f61ecfc0/torch-2.2.2-cp312-none-macosx_11_0_arm64.whl"
      sha256 "bf9558da7d2bf7463390b3b2a61a6a3dbb0b45b161ee1dd5ec640bf579d479fc"
    end
  end

  on_intel do
    resource "llvmlite" do
      url "https://files.pythonhosted.org/packages/0b/67/9443509e5d2b6d8587bae3ede5598fa8bd586b1c7701696663ea8af15b5b/llvmlite-0.43.0-cp312-cp312-macosx_10_9_x86_64.whl"
      sha256 "f99b600aa7f65235a5a05d0b9a9f31150c390f31261f2a0ba678e26823ec38f7"
    end

    resource "markupsafe" do
      url "https://files.pythonhosted.org/packages/5a/72/147da192e38635ada20e0a2e1a51cf8823d2119ce8883f7053879c2199b5/markupsafe-3.0.3-cp312-cp312-macosx_10_13_x86_64.whl"
      sha256 "d53197da72cc091b024dd97249dfc7794d6a56530370992a5e1a08983ad9230e"
    end

    resource "numba" do
      url "https://files.pythonhosted.org/packages/eb/5c/b5ec752c475e78a6c3676b67c514220dbde2725896bbb0b6ec6ea54b2738/numba-0.60.0-cp312-cp312-macosx_10_9_x86_64.whl"
      sha256 "d7da4098db31182fc5ffe4bc42c6f24cd7d1cb8a14b59fd755bfee32e34b8404"
    end

    resource "numpy" do
      url "https://files.pythonhosted.org/packages/95/12/8f2020a8e8b8383ac0177dc9570aad031a3beb12e38847f7129bacd96228/numpy-1.26.4-cp312-cp312-macosx_10_9_x86_64.whl"
      sha256 "b3ce300f3644fb06443ee2222c2201dd3a89ea6040541412b8fa189341847218"
    end

    resource "regex" do
      url "https://files.pythonhosted.org/packages/1e/95/fc7ba4303b5a0f92446a12ee6778ef2c6c799233f5060042a31bf390cfe9/regex-2026.5.9-cp312-cp312-macosx_10_13_x86_64.whl"
      sha256 "398c521292f4c7fb807001dcd54694d3a1fcafc179a36ad9cc56f98df85930b6"
    end

    resource "tiktoken" do
      url "https://files.pythonhosted.org/packages/85/8e/144bde4e01df66b34bb865557c7cd754ed08b036217ebd79c9db5e9048a9/tiktoken-0.13.0-cp312-cp312-macosx_10_13_x86_64.whl"
      sha256 "32ac870a806cfb260a02d0cb70426aef02e038297f8ad50df5040bb5af360791"
    end

    resource "torch" do
      url "https://files.pythonhosted.org/packages/79/78/29dcab24a344ffd9ee9549ec0ab2c7885c13df61cde4c65836ee275efaeb/torch-2.2.2-cp312-none-macosx_10_9_x86_64.whl"
      sha256 "eb4d6e9d3663e26cd27dc3ad266b34445a16b54908e74725adb241aa56987533"
    end
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/59/8c/57e832b7af6d7c5abe66eb3fbe3a3a32f4d11ea23a1aa7131371035be991/certifi-2026.5.20-py3-none-any.whl"
    sha256 "3c52e209ba0a4ad7aebe60436a4ab349c39e1e602e8c134221e546902ad25897"
  end

  resource "charset-normalizer" do
    url "https://files.pythonhosted.org/packages/0c/eb/4fc8d0a7110eb5fc9cc161723a34a8a6c200ce3b4fbf681bc86feee22308/charset_normalizer-3.4.7-cp312-cp312-macosx_10_13_universal2.whl"
    sha256 "eca9705049ad3c7345d574e3510665cb2cf844c2f2dcfe675332677f081cbd46"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/81/47/dd9a212ef6e343a6857485ffe25bba537304f1913bdbed446a23f7f592e1/filelock-3.29.0-py3-none-any.whl"
    sha256 "96f5f6344709aa1572bbf631c640e4ebeeb519e08da902c39a001882f30ac258"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/d5/0c/043d5e551459da400957a1395e0febbf771446ff34291afcbe3d8be2a279/fsspec-2026.4.0-py3-none-any.whl"
    sha256 "11ef7bb35dab8a394fde6e608221d5cf3e8499401c249bebaeaad760a1a8dec2"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/d2/23/408243171aa9aaba178d3e2559159c24c1171a641aa83b67bdd3394ead8e/idna-3.15-py3-none-any.whl"
    sha256 "048adeaf8c2d788c40fee287673ccaa74c24ffd8dcf09ffa555a2fbb59f10ac8"
  end

  resource "jinja2" do
    url "https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl"
    sha256 "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67"
  end

  resource "linkify-it-py" do
    url "https://files.pythonhosted.org/packages/b4/de/88b3be5c31b22333b3ca2f6ff1de4e863d8fe45aaea7485f591970ec1d3e/linkify_it_py-2.1.0-py3-none-any.whl"
    sha256 "0d252c1594ecba2ecedc444053db5d3a9b7ec1b0dd929c8f1d74dce89f86c05e"
  end

  resource "markdown-it-py" do
    url "https://files.pythonhosted.org/packages/b3/81/4da04ced5a082363ecfa159c010d200ecbd959ae410c10c0264a38cac0f5/markdown_it_py-4.2.0-py3-none-any.whl"
    sha256 "9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a"
  end

  resource "mdit-py-plugins" do
    url "https://files.pythonhosted.org/packages/a5/69/6da5581c6a7fede7dc261bf4e67d6adca4196f176b43288b55b3db395b6e/mdit_py_plugins-0.6.1-py3-none-any.whl"
    sha256 "214c82fb2ac524472ab6a5bcab1de80f73b50443e187f401bfd77efbc7c6481d"
  end

  resource "mdurl" do
    url "https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl"
    sha256 "84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8"
  end

  resource "more-itertools" do
    url "https://files.pythonhosted.org/packages/cb/98/6af411189d9413534c3eb691182bff1f5c6d44ed2f93f2edfe52a1bbceb8/more_itertools-11.0.2-py3-none-any.whl"
    sha256 "6e35b35f818b01f691643c6c611bc0902f2e92b46c18fffa77ae1e7c46e912e4"
  end

  resource "mpmath" do
    url "https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl"
    sha256 "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c"
  end

  resource "networkx" do
    url "https://files.pythonhosted.org/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl"
    sha256 "d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762"
  end

  resource "openai-whisper" do
    url "https://files.pythonhosted.org/packages/35/8e/d36f8880bcf18ec026a55807d02fe4c7357da9f25aebd92f85178000c0dc/openai_whisper-20250625.tar.gz"
    sha256 "37a91a3921809d9f44748ffc73c0a55c9f366c85a3ef5c2ae0cc09540432eb96"
  end

  resource "platformdirs" do
    url "https://files.pythonhosted.org/packages/75/a6/a0a304dc33b49145b21f4808d763822111e67d1c3a32b524a1baf947b6e1/platformdirs-4.9.6-py3-none-any.whl"
    sha256 "e61adb1d5e5cb3441b4b7710bea7e4c12250ca49439228cc1021c00dcfac0917"
  end

  resource "pygments" do
    url "https://files.pythonhosted.org/packages/f4/7e/a72dd26f3b0f4f2bf1dd8923c85f7ceb43172af56d63c7383eb62b332364/pygments-2.20.0-py3-none-any.whl"
    sha256 "81a9e26dd42fd28a23a2d169d86d7ac03b46e2f8b59ed4698fb4785f946d0176"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/a0/f4/c67b0b3f1b9245e8d266f0f112c500d50e5b4e83cb6f3b71b6528104182a/requests-2.34.2-py3-none-any.whl"
    sha256 "2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/82/3b/64d4899d73f91ba49a8c18a8ff3f0ea8f1c1d75481760df8c68ef5235bf5/rich-15.0.0-py3-none-any.whl"
    sha256 "33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb"
  end

  resource "sympy" do
    url "https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl"
    sha256 "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5"
  end

  resource "textual" do
    url "https://files.pythonhosted.org/packages/a8/f5/c1e18bc0707300a0e90204343abbf7d7acd6fb7ebe03a6d4893b99a234b8/textual-8.2.7-py3-none-any.whl"
    sha256 "4caaa13a90bc4cf9c6c862c067ccd34fe84e9c161710a2a907a8026313b6bd73"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/16/e1/3079a9ff9b8e11b846c6ac5c8b5bfb7ff225eee721825310c91b3b50304f/tqdm-4.67.3-py3-none-any.whl"
    sha256 "ee1e4c0e59148062281c49d80b25b67771a127c85fc9676d3be5f243206826bf"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl"
    sha256 "f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548"
  end

  resource "uc-micro-py" do
    url "https://files.pythonhosted.org/packages/61/73/d21edf5b204d1467e06500080a50f79d49ef2b997c79123a536d4a17d97c/uc_micro_py-2.0.0-py3-none-any.whl"
    sha256 "3603a3859af53e5a39bc7677713c78ea6589ff188d70f4fee165db88e22b242c"
  end

  resource "urllib3" do
    url "https://files.pythonhosted.org/packages/7f/3e/5db95bcf282c52709639744ca2a8b149baccf648e39c8cc87553df9eae0c/urllib3-2.7.0-py3-none-any.whl"
    sha256 "9fb4c81ebbb1ce9531cce37674bbc6f1360472bc18ca9a553ede278ef7276897"
  end

  resource "yt-dlp" do
    url "https://files.pythonhosted.org/packages/cd/13/5093bcb954878e50f7217fd2ab94282b53934022e4e4a03265582da83bf5/yt_dlp-2026.3.17-py3-none-any.whl"
    sha256 "32992db94303a8a5d211a183f2174834fe7f8c29d83ed2e7a324eae97a8f26d8"
  end

  def install
    system "python3.12", "-m", "venv", libexec
    pip = libexec/"bin/pip"

    # Homebrew caches downloads with a SHA prefix that breaks pip's wheel parser.
    # Copy each cached resource to a clean filename derived from its URL before installing.
    stage_dir = buildpath/"_resources"
    stage_dir.mkpath
    staged = resources.map do |r|
      target = stage_dir/File.basename(r.url)
      cp r.cached_download, target
      target
    end

    system pip, "install", "--no-deps", "--ignore-installed", *staged
    system pip, "install", "--no-deps", "--ignore-installed", buildpath
    bin.install_symlink libexec/"bin/ytalk"
  end

  def post_install
    Dir.glob(libexec/"lib/python*/site-packages/**/*.so").each do |so|
      next if File.symlink?(so)
      system "install_name_tool", "-id", "@rpath/#{File.basename(so)}", so
      system "codesign", "--force", "--sign", "-", so
    end
    ohai "Run 'ollama pull gemma3:4b' to download a chat model"
  end

  def caveats
    <<~EOS
      Ollama must be running for chat/summarization to work.
      Start it with: brew services start ollama
      Then pull a model: ollama pull gemma3:4b
    EOS
  end

  test do
    assert_match "usage", shell_output("#{bin}/ytalk --help")
  end
end
