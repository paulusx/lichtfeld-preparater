{ lib
, python3Packages
, colmap
, ffmpeg
, fetchurl
, linkFarm
}:

let
  # COLMAP's prebuilt FAISS vocabulary trees (COLMAP 3.12 and later read
  # them), for --matcher vocab_tree. The script picks one by image count.
  vocabTree = words: hash: {
    name = "vocab_tree_faiss_flickr100K_words${words}.bin";
    path = fetchurl {
      url = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words${words}.bin";
      inherit hash;
    };
  };
  vocabTrees = linkFarm "colmap-vocab-trees" [
    (vocabTree "32K" "sha256-kh6JS32B9c8iPfgkoCuZMmYM3fAKgVyT/HwL1pD8Y54=") # 9 MB, < 1,000 images
    (vocabTree "256K" "sha256-lsqOyOpgsfc0ZaryxAH9OzynXNui08UNai9vdg8nXdw=") # 72 MB, < 10,000
    (vocabTree "1M" "sha256-ezKbT1QcA8Q+08i5/8sg0oiGf5lfVRKA2iSvJCh4qpA=") # 282 MB, more
  ];
in
python3Packages.buildPythonApplication {
  pname = "lichtfeld-preparater";
  version = "0.1.0";
  src = ./.;
  format = "other";

  propagatedBuildInputs = [ python3Packages.typer ];

  installPhase = ''
    runHook preInstall
    install -Dm755 lichtfeld_preparater.py $out/bin/lichtfeld-preparater
    runHook postInstall
  '';

  # colmap and ffmpeg are suffixed, not prefixed: a CUDA colmap already on
  # PATH still wins, and --colmap/--ffmpeg keep overriding both.
  makeWrapperArgs = [
    "--suffix" "PATH" ":" "${lib.makeBinPath [ colmap ffmpeg ]}"
    "--set-default" "LICHTFELD_VOCAB_TREES" "${vocabTrees}"
  ];

  passthru = { inherit vocabTrees; };

  meta = {
    description = "Convert a flat image folder or a video file into a COLMAP dataset";
    homepage = "https://github.com/paulusx/lichtfeld-preparater";
    mainProgram = "lichtfeld-preparater";
    license = lib.licenses.mit;
    platforms = lib.platforms.unix;
  };
}
