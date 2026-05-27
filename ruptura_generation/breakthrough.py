import argparse
import json
import os
import random
import shutil

BASE_DIR = "break_binary"
NUM_ITERATIONS = 50000
TOTAL_ADSORBABLE_Y = 0.1
CARRIER_Y = 0.9

COLUMN_RANGES = [
    (298, 313),      # Temperature [K]
    (0.1, 1.0),      # ColumnVoidFraction [-]
    (500, 1800),     # ParticleDensity [kg/m^3]
    (0.01, 0.5),     # PressureGradient [Pa/m]
    (0.01, 0.5),     # ColumnEntranceVelocity [m/s]
    (0.1, 1.0),      # ColumnLength [m]
]

# Per component: mass-transfer coefficient, axial dispersion coefficient,
# and two Langmuir-Freundlich sites (q_sat, b, v).
PER_COMPONENT_RANGES = [
    (0.01, 0.5),      # MassTransferCoefficient [1/s]
    (0.001, 0.5),     # AxialDispersionCoefficient [m^2/s]
    (1, 20),          # LF q_sat site 1
    (1e-10, 5e-5),    # LF b site 1
    (0.5, 1.5),       # LF v site 1
    (1, 20),          # LF q_sat site 2
    (1e-10, 5e-5),    # LF b site 2
    (0.5, 1.5),       # LF v site 2
]

RANGES = COLUMN_RANGES + PER_COMPONENT_RANGES * 2
assert len(RANGES) == 22


def sample_dirichlet_simple(k: int):
    vals = [random.gammavariate(1.0, 1.0) for _ in range(k)]
    total = sum(vals)
    return [v / total for v in vals]


def build_params():
    others = [random.uniform(a, b) for a, b in RANGES]
    ys = [TOTAL_ADSORBABLE_Y * w for w in sample_dirichlet_simple(2)]

    params = [None] * 24
    params[0:6] = others[0:6]
    params[6] = ys[0]
    params[7:15] = others[6:14]
    params[15] = ys[1]
    params[16:24] = others[14:22]

    y_sum = params[6] + params[15]
    if abs(y_sum - TOTAL_ADSORBABLE_Y) > 1e-12:
        raise RuntimeError(f"adsorbable mole fraction sum mismatch: {y_sum}")
    if abs(CARRIER_Y + y_sum - 1.0) > 1e-12:
        raise RuntimeError(f"total mole fraction mismatch: {CARRIER_Y + y_sum}")
    return params


def fmt(x, nd=10):
    return f"{float(x):.{nd}f}"


def write_case(folder_name, params):
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f)

    with open(os.path.join(folder_name, "simulation.input"), "w", encoding="utf-8") as fo:
        fo.write(f"""
SimulationType           Breakthrough

// Column settings
DisplayName              DL
Temperature              {fmt(params[0], 6)}
ColumnVoidFraction       {fmt(params[1], 6)}
ParticleDensity          {fmt(params[2], 6)}
TotalPressure            1e6
PressureGradient         {fmt(params[3], 10)}
ColumnEntranceVelocity   {fmt(params[4], 10)}
ColumnLength             {fmt(params[5], 10)}

// Run settings
NumberOfTimeSteps       auto
PrintEvery              10000
WriteEvery              10000
TimeStep                0.0005
NumberOfGridPoints      100
MixturePredictionMethod SIAST

Component 0 MoleculeName               Helium
            GasPhaseMolFraction        {fmt(CARRIER_Y, 6)}
            CarrierGas                 yes

Component 1 MoleculeName               Compent1
            GasPhaseMolFraction        {fmt(params[6], 10)}
            MassTransferCoefficient    {fmt(params[7], 10)}
            AxialDispersionCoefficient {fmt(params[8], 10)}
            NumberOfIsothermSites      2
            Langmuir-Freundlich        {fmt(params[9], 10)}  {fmt(params[10], 12)}  {fmt(params[11], 10)}
            Langmuir-Freundlich        {fmt(params[12], 10)} {fmt(params[13], 12)}  {fmt(params[14], 10)}

Component 2 MoleculeName               Compent2
            GasPhaseMolFraction        {fmt(params[15], 10)}
            MassTransferCoefficient    {fmt(params[16], 10)}
            AxialDispersionCoefficient {fmt(params[17], 10)}
            NumberOfIsothermSites      2
            Langmuir-Freundlich        {fmt(params[18], 10)} {fmt(params[19], 12)} {fmt(params[20], 10)}
            Langmuir-Freundlich        {fmt(params[21], 10)} {fmt(params[22], 12)} {fmt(params[23], 10)}
""")

    if os.path.exists("run"):
        shutil.copy("run", os.path.join(folder_name, "run"))


def main():
    parser = argparse.ArgumentParser(description="Generate binary RUPTURA breakthrough simulation input folders.")
    parser.add_argument("--base_dir", default=BASE_DIR, help="Output directory for generated simulation folders.")
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--start_index", type=int, default=10001)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.base_dir, exist_ok=True)
    for i in range(args.num_iterations):
        folder_name = os.path.join(args.base_dir, f"iteration_{i + args.start_index}")
        write_case(folder_name, build_params())
    print(f"Generated {args.num_iterations} binary breakthrough simulation folders in {args.base_dir}.")


if __name__ == "__main__":
    main()
