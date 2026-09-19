{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      rust-overlay,
      treefmt-nix,
      ...
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f (
            let
              pkgs = import nixpkgs {
                inherit system;
                overlays = [ (import rust-overlay) ];
                config.allowUnfree = true;
              };

              rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

              rustPlatform = pkgs.makeRustPlatform {
                rustc = rust;
                cargo = rust;
              };

              formatter =
                (treefmt-nix.lib.evalModule pkgs {
                  projectRootFile = "flake.nix";

                  settings = {
                    allow-missing-formatter = true;
                    verbose = 0;

                    global.excludes = [ "*.lock" ];

                    formatter = {
                      nixfmt.options = [ "--strict" ];

                      rustfmt = {
                        package = rust;

                        options = [
                          "--config-path"
                          (toString ./rustfmt.toml)
                        ];
                      };
                    };
                  };

                  programs = {
                    nixfmt.enable = true;
                    oxfmt.enable = true;
                    rustfmt = {
                      enable = true;
                      package = rust;
                    };
                    taplo.enable = true;
                  };
                }).config.build.wrapper;

              # Only what cargo reads goes into the derivation hash, so
              # a stray `target/` or `.jj/` write never forces a
              # rebuild.
              rustSrc = pkgs.lib.fileset.toSource {
                root = ./.;
                fileset = pkgs.lib.fileset.unions [
                  ./Cargo.toml
                  ./Cargo.lock
                  ./rust-toolchain.toml
                  ./rustfmt.toml
                  ./deny.toml
                  ./src
                  ./benches
                ];
              };

              # Builds the `vs1` binary (and library) as a Nix package.
              # Pass `name` plus `buildFeatures` / `buildInputs` /
              # `extraEnv` to opt into `cuda` / `metal`.
              mkPackage =
                {
                  name,
                  buildFeatures ? [ ],
                  buildInputs ? [ ],
                  nativeBuildInputs ? [ ],
                  extraEnv ? { },
                }:
                rustPlatform.buildRustPackage (
                  {
                    inherit name buildFeatures buildInputs;
                    # `remove-references-to` strips the rust toolchain
                    # path that rustc bakes into binary debug info via
                    # `rust-src`; without it the runtime closure drags
                    # in the whole nightly toolchain.
                    nativeBuildInputs = nativeBuildInputs ++ [ pkgs.removeReferencesTo ];
                    src = rustSrc;
                    doCheck = false;
                    cargoLock.lockFile = ./Cargo.lock;
                    RUSTFLAGS = "-C target-cpu=native";
                    meta.mainProgram = "vs1";
                    postInstall = ''
                      for bin in "$out"/bin/*; do
                        remove-references-to -t ${rust} "$bin"
                      done
                    '';
                    disallowedReferences = [ rust ];
                  }
                  // extraEnv
                );
            in
            {
              inherit
                formatter
                mkPackage
                pkgs
                rust
                system
                ;
            }
          )
        );
    in
    {
      packages = forEachSupportedSystem (
        { mkPackage, pkgs, ... }:
        let
          cudaNativeBuildInputs = with pkgs; [
            cudaPackages.cuda_nvcc
            autoAddDriverRunpath
          ];
          cudaBuildInputs = with pkgs.cudaPackages; [
            cuda_nvcc
            cudatoolkit
            cudnn
          ];
          cudaEnv = {
            CUDA_COMPUTE_CAP = "80";
            CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          };
        in
        {
          default = mkPackage { name = "vs1"; };

          vs1 = mkPackage { name = "vs1"; };

          vs1-cuda = mkPackage {
            name = "vs1-cuda";
            buildFeatures = [ "cuda" ];
            nativeBuildInputs = cudaNativeBuildInputs;
            buildInputs = cudaBuildInputs;
            extraEnv = cudaEnv;
          };

          vs1-metal = mkPackage {
            name = "vs1-metal";
            buildFeatures = [ "metal" ];
          };
        }
      );

      formatter = forEachSupportedSystem ({ formatter, ... }: formatter);

      devShells = forEachSupportedSystem (
        {
          pkgs,
          rust,
          formatter,
          ...
        }:
        {
          default = pkgs.mkShell (
            {
              name = "vs1";

              buildInputs =
                with pkgs;
                [
                  formatter
                  rust

                  bacon
                  cargo-deny
                  cargo-nextest
                  uv
                ]
                ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux (
                  with pkgs.cudaPackages;
                  [
                    cuda_nvcc
                    cudatoolkit
                    cudnn
                  ]
                );
            }
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              CUDA_COMPUTE_CAP = "80";
              CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";

              shellHook = ''
                # CUDA binaries built here need the driver's libcuda, not
                # the toolkit's stub, at run time:
                # export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              '';
            }
          );
        }
      );
    };
}
