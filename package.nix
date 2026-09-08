{ lib
, python3Packages
, colmap
, ffmpeg
}:

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
  ];

  meta = {
    description = "Convert a flat image folder or a video file into a COLMAP dataset";
    homepage = "https://github.com/paulusx/lichtfeld-preparater";
    mainProgram = "lichtfeld-preparater";
    license = lib.licenses.mit;
    platforms = lib.platforms.unix;
  };
}
