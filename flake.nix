
{
  description = "OHLC Volatility Model";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          python-dotenv
          requests
          pip
          virtualenv
          pandas
          numpy
          matplotlib
          seaborn
          scipy
          jupyterlab
          ipykernel
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.gnumake
          ];

          shellHook = ''
            echo "loaded"

            if [ ! -d .venv ]; then
              echo "creating venv"
              python -m venv --system-site-packages .venv
            fi

            source .venv/bin/activate

            pip install polars pyarrow duckdb databento mplfinance exchange-calendars --quiet
            pip install -e . --quiet

            echo "Python ${pkgs.python311.version} ready"
          '';
        };
      }
    );
}
