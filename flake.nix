{
  description = "lichtfeld-preparater — turn an image folder or a video into a COLMAP dataset";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: rec {
        lichtfeld-preparater = pkgs.python3Packages.buildPythonApplication {
          pname = "lichtfeld-preparater";
          version = "0.1.0";
          src = ./.;
          format = "other";

          propagatedBuildInputs = [ pkgs.python3Packages.typer ];

          installPhase = ''
            runHook preInstall
            install -Dm755 lichtfeld_preparater.py $out/bin/lichtfeld-preparater
            runHook postInstall
          '';

          # colmap and ffmpeg are suffixed, not prefixed: a CUDA colmap already on
          # PATH still wins, and --colmap/--ffmpeg keep overriding both.
          makeWrapperArgs = [
            "--suffix" "PATH" ":" "${pkgs.lib.makeBinPath [ pkgs.colmap pkgs.ffmpeg ]}"
          ];

          meta = with pkgs.lib; {
            description = "Convert a flat image folder or a video file into a COLMAP dataset";
            homepage = "https://github.com/paulusx/lichtfeld-preparater";
            mainProgram = "lichtfeld-preparater";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };

        default = lichtfeld-preparater;
      });

      apps = forAllSystems (pkgs: rec {
        lichtfeld-preparater = {
          type = "app";
          program = "${self.packages.${pkgs.system}.lichtfeld-preparater}/bin/lichtfeld-preparater";
        };
        default = lichtfeld-preparater;
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: [ ps.typer ]))
            pkgs.colmap
            pkgs.ffmpeg
          ];
        };
      });

      overlays.default = final: prev: {
        lichtfeld-preparater = self.packages.${final.system}.lichtfeld-preparater;
      };

      formatter = forAllSystems (pkgs: pkgs.nixpkgs-fmt);
    };
}
