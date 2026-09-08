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
        # package.nix is callPackage-shaped on purpose: channel-based configs can
        # import it directly and swap in their own colmap (e.g. a CUDA build).
        lichtfeld-preparater = pkgs.callPackage ./package.nix { };

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
