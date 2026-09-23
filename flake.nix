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
    # candle-flash-attn's build script (via cudaforge) clones this
    # exact commit of cutlass at build time. Pinning it as a flake
    # input lets the sandboxed `-cuda` build stage a copy
    # instead of reaching for the network.
    nvidia-cutlass = {
      url = "github:NVIDIA/cutlass/7d49e6c7e2f8896c47f586706e67e1fb215529dc";
      flake = false;
    };
  };

  outputs =
    {
      nixpkgs,
      nvidia-cutlass,
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
                  ./crates/vs1/Cargo.toml
                  ./crates/vs1/build.rs
                  ./crates/vs1/src
                  ./crates/vs1/benches
                  ./crates/vs1/tests
                  ./crates/vs1-browser/Cargo.toml
                  ./crates/vs1-browser/src
                  ./crates/vs1-browser/assets
                  ./crates/vs1-email
                  ./examples/email-rules.toml
                ];
              };

              # Builds one workspace CLI as a Nix package.
              # Pass `name` plus `buildFeatures` / `buildInputs` /
              # `extraEnv` / `extraPreBuild` to opt into `cuda` /
              # `metal`.
              mkPackage =
                {
                  name,
                  crate ? "vs1",
                  buildFeatures ? [ ],
                  buildInputs ? [ ],
                  nativeBuildInputs ? [ ],
                  extraEnv ? { },
                  extraPreBuild ? "",
                }:
                rustPlatform.buildRustPackage (
                  {
                    inherit name buildInputs;
                    buildFeatures = map (feature: "${crate}/${feature}") buildFeatures;
                    cargoBuildFlags = [
                      "-p"
                      crate
                    ];
                    cargoTestFlags = [
                      "-p"
                      crate
                    ];
                    # `remove-references-to` strips the rust toolchain
                    # path that rustc bakes into binary debug info via
                    # `rust-src`; without it the runtime closure drags
                    # in the whole nightly toolchain.
                    nativeBuildInputs = nativeBuildInputs ++ [ pkgs.removeReferencesTo ];
                    src = rustSrc;
                    doCheck = false;
                    cargoLock.lockFile = ./Cargo.lock;
                    RUSTFLAGS = "-C target-cpu=native";
                    meta.mainProgram = crate;
                    preBuild = extraPreBuild;
                    postInstall = ''
                      for bin in "$out"/bin/*; do
                        remove-references-to -t ${rust} "$bin"
                      done
                    '';
                    # Guard: fail the build if a rust toolchain reference
                    # survives, so this closure leak can't silently return.
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
          # cudaforge fetches NVIDIA/cutlass via git at build time.
          # Pre-stage a sandbox-resident copy with a stubbed `.git` so
          # the build doesn't need network and `git rev-parse HEAD`
          # returns the pinned commit.
          cudaforgeEnv = cudaEnv // {
            CUDAFORGE_HOME = "/tmp/cudaforge-cache";
          };
          cudaforgePreBuild = ''
            dest=$CUDAFORGE_HOME/git/checkouts/cutlass-7d49e6c7e2f8896c
            mkdir -p $CUDAFORGE_HOME/git/checkouts
            cp -r ${nvidia-cutlass} $dest
            chmod -R u+w $dest

            # Stub a minimal .git dir so cudaforge's `git rev-parse HEAD`
            # returns the expected commit hash and skips any network fetch.
            mkdir -p $dest/.git/objects $dest/.git/refs
            echo "7d49e6c7e2f8896c47f586706e67e1fb215529dc" > $dest/.git/HEAD
          '';
        in
        {
          default = mkPackage { name = "vs1"; };
          vs1-browser-local = mkPackage {
            name = "vs1-browser-local";
            crate = "vs1-browser";
            buildFeatures = [ "local" ];
          };
        }
        //
          pkgs.lib.concatMapAttrs
            (crate: _: {
              ${crate} = mkPackage {
                name = crate;
                inherit crate;
              };
              "${crate}-cuda" = mkPackage {
                name = "${crate}-cuda";
                inherit crate;
                buildFeatures = [ "cuda" ];
                nativeBuildInputs = cudaNativeBuildInputs ++ [ pkgs.git ];
                buildInputs = cudaBuildInputs;
                extraEnv = cudaforgeEnv;
                extraPreBuild = cudaforgePreBuild;
              };
              "${crate}-metal" = mkPackage {
                name = "${crate}-metal";
                inherit crate;
                buildFeatures = [ "metal" ];
              };
            })
            {
              vs1 = null;
              vs1-browser = null;
              vs1-email = null;
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
                  cargo-mutants
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
