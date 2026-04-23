{
  description = "OHLC Excursion Model";

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
          scikit-learn
          mplfinance
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

            pip install polars pyarrow yfinance duckdb xgboost optuna exchange-calendars fredapi databento pytest beautifulsoup4 holidays --quiet
            pip install -e . --quiet

            python -m ipykernel install --user --name ohlc-excursion-model --display-name "ohlc-excursion-model"

            echo "Python ${pkgs.python311.version} ready"
          '';
        };
      }
    );
}