import numpy as np
from pyDOE import lhs
from scipy.optimize import differential_evolution
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import dataclasses

@dataclass
class EnvironmentalConditions:
    temperature: float  # °C
    co2_concentration: float  # ppm
    light_intensity: float  # μmol/m²/s (PPFD)
    relative_humidity: float  # %
    wind_speed: float  # m/s
    soil_moisture: float  # % of field capacity
    soil_nutrients: float  # arbitrary 0-1 scale
    atmospheric_pressure: float = 101.325  # kPa
    ozone_concentration: float = 40  # ppb
    soil_nitrogen: float = 0.8  # 0-1 scale (NH4+ + NO3- availability)
    soil_phosphorus: float = 0.7  # 0-1 scale
    soil_pH: float = 6.5  # pH value
    UVB_intensity: float = 2.0  # W/m²

    def calculate_vpd(self) -> float:
        """Calculate vapor pressure deficit (kPa)"""
        svp = 0.6108 * np.exp(17.27 * self.temperature / (self.temperature + 237.3))
        avp = svp * (self.relative_humidity / 100)
        return max(svp - avp, 0.05)  # Prevent division by zero


class PlantPhysiology:
    def __init__(self, params):
        # Canopy structure parameters
        self.leaf_mass_per_area = params['leaf_mass_per_area']  # kg/m²
        self.specific_leaf_area = params['specific_leaf_area']  # m²/kg
        self.leaf_angle_distribution = params['leaf_angle_distribution']  # degrees
        self.leaf_thickness = params['leaf_thickness']  # mm
        self.leaf_number = params['leaf_number']  # count
        self.avg_leaf_size = params['avg_leaf_size']  # m² per leaf

        # Biochemical parameters
        self.vcmax25 = params['vcmax25']  # μmol/m²/s
        self.jmax25 = params['jmax25']  # μmol/m²/s
        self.tpmu25 = params['tpmu25']  # μmol/m²/s
        self.rd25 = params['rd25']  # μmol/m²/s
        self.gm25 = params['gm25']  # mesophyll conductance at 25°C (mol/m²/s)

        # Stomatal parameters
        self.g0 = params['g0']  # minimum conductance (mol/m²/s)
        self.g1 = params['g1']  # Medlyn model slope
        self.stomatal_density = params['stomatal_density']  # stomata/mm²

        # Photosynthetic apparatus
        self.chl_a_b_ratio = params['chl_a_b_ratio']
        self.total_chlorophyll = params['total_chlorophyll']  # mg/g
        self.rubisco_content = params['rubisco_content']  # g/m²

        # Root system
        self.root_mass_ratio = params['root_mass_ratio']
        self.specific_root_length = params['specific_root_length']  # m/g
        self.root_hydraulic_conductance = params['root_hydraulic_conductance']  # mmol/m²/s/MPa

        # New canopy parameters
        self.leaf_lifespan = params['leaf_lifespan']  # days
        self.phyllotaxis = params['phyllotaxis']  # 0-1 (spiral arrangement)
        self.leaf_pubescence = params['leaf_pubescence']  # 0-1 (hair density)

        # New photosynthetic parameters
        self.cyclic_electron_flow = params['cyclic_electron_flow']  # 0-1 ratio
        self.rubisco_activation = params['rubisco_activation']  # 0-1
        self.chloroplast_movement = params['chloroplast_movement']  # 0-1

        # New stomatal parameters
        self.stomatal_response_time = params['stomatal_response_time']  # minutes
        self.stomatal_closure_threshold = params['stomatal_closure_threshold']  # MPa

        # New root parameters
        self.root_exudation_rate = params['root_exudation_rate']  # mmol C/g root/day
        self.mycorrhizal_colonization = params['mycorrhizal_colonization']  # 0-1

        # New stress response parameters
        self.heat_shock_proteins = params['heat_shock_proteins']  # 0-1
        self.antioxidant_capacity = params['antioxidant_capacity']  # mmol ASA/g
        self.cold_hardening = params['cold_hardening']  # 0-1

        # New hydraulic parameters
        self.xylem_conductivity = params['xylem_conductivity']  # kg/m/MPa/s
        self.embolism_repair_rate = params['embolism_repair_rate']  # %/hour

        # New nutrient parameters
        self.nitrate_reductase = params['nitrate_reductase']  # μmol NO3-/g/h
        self.phosphatase_activity = params['phosphatase_activity']  # μmol PO4³⁻/g/h

    def temperature_response(self, rate25, temp, ea, ds=0, hd=0):
        """Enhanced Arrhenius temperature response with deactivation"""
        R = 8.314  # J/mol/K
        TK = temp + 273.15
        TK25 = 298.15

        arrhenius = np.exp((ea * (TK - TK25)) / (R * TK * TK25))
        peak_factor = (1 + np.exp((ds * TK25 - hd) / (R * TK25))) / \
                      (1 + np.exp((ds * TK - hd) / (R * TK)))

        return rate25 * arrhenius * peak_factor

    def calculate_photosynthesis(self, conditions: EnvironmentalConditions) -> float:
        # Temperature adjustments
        temp = conditions.temperature
        vcmax = self.temperature_response(self.vcmax25, temp, 58520, 649.1, 200000)
        jmax = self.temperature_response(self.jmax25, temp, 35870, 643.9, 200000)
        rd = self.temperature_response(self.rd25, temp, 46390)
        gm = self.temperature_response(self.gm25, temp, 46000)

        # Light absorption
        lai = self.calculate_lai()
        absorbed_ppfd = self.calculate_light_absorption(conditions, lai)

        # Iterative solution for A and gs
        A_net, ci, gs = self.solve_photosynthesis(
            conditions, vcmax, jmax, rd, gm, absorbed_ppfd
        )

        ozone_limitation = self._calculate_ozone_effect(conditions)
        nutrient_limitation = self._calculate_nutrient_limitation_v2(conditions)
        UV_protection = self._calculate_UV_protection(conditions)

        A_net *= ozone_limitation * nutrient_limitation * UV_protection
        return max(A_net, 0)  # Prevent negative photosynthesis

    def _calculate_ozone_effect(self, conditions):
        """Ozone damage to photosynthesis apparatus"""
        base_damage = 0.05 * conditions.ozone_concentration
        protection = self.antioxidant_capacity * 0.2
        return 1 - (base_damage - protection) / 100

    def _calculate_nutrient_limitation_v2(self, conditions):
        """Improved nutrient limitation considering multiple nutrients"""
        N_effect = 1 - np.exp(-5 * conditions.soil_nitrogen * self.nitrate_reductase)
        P_effect = 1 - np.exp(-3 * conditions.soil_phosphorus * self.phosphatase_activity)
        pH_effect = np.exp(-0.5 * (conditions.soil_pH - 6.0) ** 2)
        return min(N_effect, P_effect) * pH_effect

    def _calculate_UV_protection(self, conditions):
        """UV-B protection through pubescence and antioxidants"""
        pubescence_protection = self.leaf_pubescence * 0.7
        antioxidant_protection = self.antioxidant_capacity * 0.3
        UV_damage = conditions.UVB_intensity * (1 - pubescence_protection - antioxidant_protection)
        return 1 - UV_damage / 10

    def calculate_transpiration(self, conditions):
        """Penman-Monteith based transpiration"""
        VPD = conditions.calculate_vpd()
        gb = self.calculate_boundary_conductance(conditions.wind_speed)
        gc = 1 / (1 / self.stomatal_conductance + 1 / gb)
        return (0.622 * VPD * gc) / (conditions.atmospheric_pressure * 1000)

    def _calculate_water_stress_v2(self, conditions):
        """Improved water stress with hydraulic limitations"""
        soil_water_potential = -1.5 * (1 - conditions.soil_moisture)
        root_water_uptake = self.root_hydraulic_conductance * (soil_water_potential + 2)
        xylem_safety = 1 / (1 + np.exp(5 * (soil_water_potential + self.stomatal_closure_threshold)))
        return xylem_safety * (1 - np.exp(-root_water_uptake))

    def solve_photosynthesis(self, conditions, vcmax, jmax, rd, gm, absorbed_ppfd):
        """Iterative solution for coupled photosynthesis-stomatal conductance"""
        max_iter = 50
        tolerance = 0.1  # μmol/m²/s
        Ca = conditions.co2_concentration
        cs = Ca  # Initial assumption for surface CO2
        gs = self.g0 + 0.1  # Initial guess
        A_prev = 0

        for _ in range(max_iter):
            # Calculate intercellular CO2 (ci)
            gb = self.calculate_boundary_conductance(conditions.wind_speed)
            ci = cs - (A_prev / (1.6 * (1 / (1 / gs + 1 / gb))))

            # Chloroplast CO2 (cc)
            cc = ci - (A_prev / gm) if gm > 0 else ci

            # Photosynthesis limitations
            Ac = self.calculate_rubisco_limited(vcmax, cc, conditions)
            Aj = self.calculate_light_limited(jmax, cc, absorbed_ppfd, conditions)
            Ap = self.calculate_tpu_limited(cc, conditions.temperature)
            A_gross = min(Ac, Aj, Ap)
            A_net = A_gross - rd

            # Update stomatal conductance using Medlyn model
            vpd = conditions.calculate_vpd() * 1000  # kPa to Pa
            gs_new = self.g0 + 1.6 * (1 + self.g1 / np.sqrt(vpd)) * (A_net / cs)

            # Convergence check
            if abs(A_net - A_prev) < tolerance and abs(gs_new - gs) < 0.01:
                break

            gs = max(gs_new, self.g0)
            A_prev = A_net

        return A_net, ci, gs

    def calculate_rubisco_limited(self, vcmax, cc, conditions):
        """Rubisco-limited CO2 assimilation"""
        Kc = self.temperature_response(404.9, conditions.temperature, 79430)
        Ko = self.temperature_response(278.4, conditions.temperature, 36380)
        gamma_star = self.temperature_response(42.75, conditions.temperature, 37830)
        o2 = 210  # mmol/mol

        return vcmax * (cc - gamma_star) / (cc + Kc * (1 + o2 / Ko))

    def calculate_light_limited(self, jmax, cc, absorbed_ppfd, conditions):
        """Electron transport-limited assimilation"""
        gamma_star = self.temperature_response(42.75, conditions.temperature, 37830)
        J = (0.7 * absorbed_ppfd * self.quantum_yield * jmax) / \
            np.sqrt((0.7 * absorbed_ppfd * self.quantum_yield) ** 2 + jmax ** 2)
        return J * (cc - gamma_star) / (4 * (cc + 2 * gamma_star))

    def calculate_tpu_limited(self, cc, temp):
        """Triose phosphate utilization limitation"""
        tpu = self.temperature_response(self.tpmu25, temp, 53100)
        return 3 * tpu

    def calculate_lai(self) -> float:
        """Calculate leaf area index from structural parameters"""
        return min(self.leaf_mass_per_area * self.specific_leaf_area, 10)  # Cap LAI at 10

    def calculate_light_absorption(self, conditions: EnvironmentalConditions, lai: float) -> float:
        """Modified Beer-Lambert law with canopy light extinction"""
        k = 0.5 / np.sin(np.radians(self.leaf_angle_distribution))
        return conditions.light_intensity * (1 - np.exp(-k * lai))

    def calculate_boundary_conductance(self, wind_speed: float) -> float:
        """Boundary layer conductance for CO2 (mol/m²/s)"""
        # Empirical relationship for laminar flow over leaves
        return 0.147 * np.sqrt(wind_speed / 0.1) * 0.08  # Convert to mol/m²/s

    @property
    def quantum_yield(self) -> float:
        """Quantum yield efficiency based on chlorophyll content"""
        return 0.85 * (1 - np.exp(-0.05 * self.total_chlorophyll))

    def _calculate_water_stress(self, conditions: EnvironmentalConditions) -> float:
        """Water stress factor considering root hydraulic properties"""
        soil_water_potential = -np.log(conditions.soil_moisture)  # Simplified relationship
        root_water_uptake = self.root_hydraulic_conductance * soil_water_potential
        return 1 / (1 + np.exp(-5*(root_water_uptake - 0.5)))  # Sigmoidal response

    def _calculate_light_absorption(self, conditions):
        # Beer-Lambert law with leaf angle distribution
        k = 0.5 / np.sin(np.radians(self.leaf_angle_distribution))  # extinction coefficient
        lai = self.specific_leaf_area * self.total_chlorophyll  # effective leaf area index
        return conditions.light_intensity * (1 - np.exp(-k * lai))

    def _calculate_nutrient_limitation(self, conditions):
        # Simple nutrient limitation factor
        return 1 - np.exp(-2 * conditions.soil_nutrients)

    @property
    def quantum_yield(self):
        """Calculate quantum yield based on chlorophyll content"""
        return 0.85 * (1 - np.exp(-0.5 * self.total_chlorophyll))


def create_parameter_space():
    """Define the complete parameter space with all parameters"""
    param_space = {
        # Canopy parameters
        'leaf_mass_per_area': (0.1, 5.0),  # kg/m²
        'specific_leaf_area': (10, 50),  # m²/kg
        'leaf_angle_distribution': (1, 85),  # degrees
        'leaf_thickness': (0.1, 0.5),  # mm
        'leaf_number': (5, 500),  # count
        'avg_leaf_size': (0.001, 0.1),  # m²

        # Biochemical parameters
        'vcmax25': (20, 200),  # μmol/m²/s
        'jmax25': (40, 400),  # μmol/m²/s
        'tpmu25': (5, 50),  # μmol/m²/s
        'rd25': (0.5, 5.0),  # μmol/m²/s
        'gm25': (0.1, 5.0),  # mol/m²/s

        # Stomatal parameters
        'g0': (0.005, 0.1),  # mol/m²/s
        'g1': (1, 15),  # kPa^0.5
        'stomatal_density': (50, 1000),  # stomata/mm²

        # Photosynthetic apparatus
        'chl_a_b_ratio': (2.0, 4.0),
        'total_chlorophyll': (0.2, 2.0),  # mg/g
        'rubisco_content': (0.5, 5.0),  # g/m²

        # Root system
        'root_mass_ratio': (0.05, 0.5),
        'specific_root_length': (50, 500),  # m/g
        'root_hydraulic_conductance': (0.1, 10.0),  # mmol/m²/s/MPa

        # New parameters
        'leaf_lifespan': (10, 365),
        'phyllotaxis': (0.25, 0.75),
        'leaf_pubescence': (0.0, 1.0),
        'cyclic_electron_flow': (0.1, 0.9),
        'rubisco_activation': (0.5, 1.0),
        'chloroplast_movement': (0.0, 1.0),
        'stomatal_response_time': (5, 60),
        'stomatal_closure_threshold': (-3.0, -0.5),
        'root_exudation_rate': (0.01, 1.0),
        'mycorrhizal_colonization': (0.0, 0.9),
        'heat_shock_proteins': (0.0, 1.0),
        'antioxidant_capacity': (0.1, 5.0),
        'cold_hardening': (0.0, 1.0),
        'xylem_conductivity': (1e-5, 1e-3),
        'embolism_repair_rate': (0.1, 10.0),
        'nitrate_reductase': (0.1, 10.0),
        'phosphatase_activity': (0.1, 5.0)
    }
    return param_space


def create_test_conditions(base_conditions):
    """Create a list of test conditions for evaluating plant performance"""
    test_conditions = []

    # Temperature variations (more points)
    for temp in np.linspace(10, 40, 10):
        test_conditions.append(dataclasses.replace(base_conditions, temperature=temp))

    # Light intensity variations (more points)
    for light in np.linspace(100, 2000, 10):
        test_conditions.append(dataclasses.replace(base_conditions, light_intensity=light))

    # CO2 variations
    for co2 in [200, 400, 800, 1200]:
        test_conditions.append(dataclasses.replace(base_conditions, co2_concentration=co2))

    # Humidity variations
    for rh in [20, 40, 60, 80]:
        test_conditions.append(dataclasses.replace(base_conditions, relative_humidity=rh))

    # Soil moisture variations
    for soil_moisture in np.linspace(0.2, 1.0, 5):
        test_conditions.append(dataclasses.replace(base_conditions, soil_moisture=soil_moisture))

    return test_conditions


def evaluate_plant(params_dict, test_conditions):
    """Evaluate plant performance across all test conditions"""
    plant = PlantPhysiology(params_dict)
    rates = []
    weights = []

    for condition in test_conditions:
        try:
            rate = plant.calculate_photosynthesis(condition)

            # Add weight based on condition frequency in nature
            weight = 1.0
            if 15 <= condition.temperature <= 30:  # Optimal temperature range
                weight *= 1.5
            if 400 <= condition.co2_concentration <= 800:  # Current to near-future CO2
                weight *= 1.2
            if condition.light_intensity >= 1000:  # Full sunlight
                weight *= 1.3

            rates.append(rate)
            weights.append(weight)

        except Exception as e:
            print(f"Warning: Error calculating photosynthesis: {e}")
            rates.append(-1000)
            weights.append(0.1)

    return -np.average(rates, weights=weights)  # Negative because we want to maximize


def objective_function(x, param_names, test_conditions):
    params = dict(zip(param_names, x))

    # Nitrogen allocation constraint (Vcmax + Jmax + Rubisco)
    if params['vcmax25'] * 0.8 + params['jmax25'] * 0.4 > 150:  # Arbitrary nitrogen budget
        return 1000  # Heavy penalty

    # Carbon cost for thick leaves
    leaf_cost = params['leaf_thickness'] * params['specific_leaf_area'] * 0.2
    if leaf_cost > 15:  # Simulated carbon budget
        return 1000

    return evaluate_plant(params, test_conditions)


def optimize_plant(n_samples=10000000, n_generations=500):
    param_space = create_parameter_space()
    param_names = list(param_space.keys())
    bounds = [param_space[param] for param in param_names]

    base_conditions = EnvironmentalConditions(
        temperature=25,
        co2_concentration=400,
        light_intensity=1500,
        relative_humidity=60,
        wind_speed=2,
        soil_moisture=0.8,
        soil_nutrients=0.9
    )

    test_conditions = create_test_conditions(base_conditions)

    try:
        print("Starting optimization with parallel processing...")
        result = differential_evolution(
            objective_function,  # Use the top-level function
            bounds,
            args=(param_names, test_conditions),  # Pass required parameters
            maxiter=n_generations,
            popsize=50,
            mutation=(0.5, 1.5),
            recombination=0.9,
            updating='deferred',
            workers=4
        )
        print("Optimization complete!")
    except Exception as e:
        print(f"Optimization error: {e}")
        return None, None, None

    optimal_params = dict(zip(param_names, result.x))
    optimal_plant = PlantPhysiology(optimal_params)

    return optimal_plant, optimal_params, result


def analyze_sensitivity(optimal_params, n_samples=5000):
    param_space = create_parameter_space()
    results = []

    # Create base conditions
    base_conditions = EnvironmentalConditions(
        temperature=25,
        co2_concentration=400,
        light_intensity=1500,
        relative_humidity=60,
        wind_speed=2,
        soil_moisture=0.8,
        soil_nutrients=0.9
    )

    # Create test matrix (fewer conditions for faster analysis)
    test_conditions = []
    for temp in [15, 25, 35]:
        for light in [500, 1500, 2500]:
            for co2 in [400, 800]:
                test_conditions.append(dataclasses.replace(base_conditions,
                                                           temperature=temp,
                                                           light_intensity=light,
                                                           co2_concentration=co2
                                                           ))

    print(f"Analyzing {len(param_space)} parameters across {len(test_conditions)} conditions...")

    for param_name in optimal_params.keys():
        print(f"Analyzing sensitivity for {param_name}...")
        param_range = param_space[param_name]
        values = np.linspace(param_range[0], param_range[1], n_samples)
        rates_matrix = []

        for value in values:
            test_params = optimal_params.copy()
            test_params[param_name] = value
            plant = PlantPhysiology(test_params)

            condition_rates = []
            for condition in test_conditions:
                rate = plant.calculate_photosynthesis(condition)
                condition_rates.append(rate)
            rates_matrix.append(condition_rates)

        rates_matrix = np.array(rates_matrix)
        sensitivity = np.mean([np.std(rates_matrix[:, i]) / np.mean(rates_matrix[:, i])
                               for i in range(rates_matrix.shape[1])])

        results.append({
            'parameter': param_name,
            'sensitivity': sensitivity,
            'values': values,
            'rates': np.mean(rates_matrix, axis=1)
        })

    return pd.DataFrame(results)


def plot_results(sensitivity_results):
    plt.figure(figsize=(15, 10))

    # Parameter sensitivities
    plt.subplot(2, 1, 1)
    sns.barplot(data=sensitivity_results, x='parameter', y='sensitivity')
    plt.xticks(rotation=45, ha='right')
    plt.title('Parameter Sensitivities')

    # Response curves for top 3 parameters
    plt.subplot(2, 1, 2)
    top_params = sensitivity_results.nlargest(3, 'sensitivity')
    for _, row in top_params.iterrows():
        plt.plot(row['values'], row['rates'], label=row['parameter'])
    plt.xlabel('Parameter Value')
    plt.ylabel('Photosynthesis Rate')
    plt.title('Response Curves for Top 3 Parameters')
    plt.legend()

    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    try:
        print("Starting optimization...")
        result = optimize_plant()

        if result[0] is None:
            print("Optimization failed!")
        else:
            optimal_plant, optimal_params, optimization_result = result
            print("Analyzing sensitivities...")
            sensitivity_results = analyze_sensitivity(optimal_params)
            print("Plotting results...")
            plot_results(sensitivity_results)
            plt.show()
            print("\nOptimal Plant Parameters:")
            for param, value in optimal_params.items():
                print(f"{param}: {value:.3f}")

    except Exception as e:
        print(f"An error occurred: {e}")