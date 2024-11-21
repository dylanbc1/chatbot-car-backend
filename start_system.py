from experta import *
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import logging

logging.getLogger("experta.watchers").setLevel(logging.ERROR)

class StartingInference:
    def __init__(self):
        self.model = self._build_model()
        self.inference = VariableElimination(self.model)

    def _build_model(self):
        
        model = BayesianNetwork([
            ('StarterSystem', 'starter_cranks'),
            ('StarterSystem', 'starter_spins'),
            ('BatterySystem', 'battery_voltage'),
            ('BatterySystem', 'cleaned_terminals'),
            ('BatterySystem', 'starter_cranks'),
            ('BatterySystem', 'IgnitionSystem'),
            ('FuelSystem', 'fuel_to_filter'),
            ('FuelSystem', 'fuel_to_injector'),
            ('FuelSystem', 'starts_and_stalls'),
            ('FuelSystem', 'engine_fires'),
            ('IgnitionSystem', 'spark_to_plugs'),
            ('IgnitionSystem', 'spark_from_coil'),
            ('IgnitionSystem', 'coil_primary_voltage'),
            ('IgnitionSystem', 'mechanical_distributor'),
            ('IgnitionSystem', 'engine_fires'),
            ('SensorSystem', 'obd_codes'),
            ('SensorSystem', 'stalls_on_key_release'),
            ('SensorSystem', 'stalls_in_rain'),
            ('SensorSystem', 'stalls_when_warm'),
            ('SensorSystem', 'stalls_when_cold')
        ])

        cpd_starter = TabularCPD(
            variable='StarterSystem', variable_card=2,
            values=[[0.75], [0.25]]
        )

        cpd_battery = TabularCPD(
            variable='BatterySystem', variable_card=2,
            values=[[0.85], [0.15]]
        )

        cpd_fuel = TabularCPD(
            variable='FuelSystem', variable_card=2,
            values=[[0.90], [0.10]]
        )
        
        cpd_ignition = TabularCPD(
            variable='IgnitionSystem', 
            variable_card=2,
            evidence=['BatterySystem'],
            evidence_card=[2],
            values=[
                [0.90, 0.30],
                [0.10, 0.70]
            ]
        )
        
        cpd_sensor = TabularCPD(
            variable='SensorSystem', variable_card=2,
            values=[[0.95], [0.05]]
        )

        cpd_starter_cranks = TabularCPD(
            variable='starter_cranks', variable_card=2,
            evidence=['StarterSystem', 'BatterySystem'], evidence_card=[2, 2],
            values=[[0.99, 0.70, 0.80, 0.10],
                    [0.01, 0.30, 0.20, 0.90]]
        )

        cpd_starter_spins = TabularCPD(
            variable='starter_spins', variable_card=2,
            evidence=['StarterSystem'], evidence_card=[2],
            values=[[0.95, 0.20],
                    [0.05, 0.80]]
        )

        cpd_battery_voltage = TabularCPD(
            variable='battery_voltage', variable_card=2,
            evidence=['BatterySystem'], evidence_card=[2],
            values=[[0.98, 0.30],
                    [0.02, 0.70]]
        )

        cpd_cleaned_terminals = TabularCPD(
            variable='cleaned_terminals', variable_card=2,
            evidence=['BatterySystem'], evidence_card=[2],
            values=[[0.90, 0.40],
                    [0.10, 0.60]]
        )

        cpd_fuel_to_filter = TabularCPD(
            variable='fuel_to_filter', variable_card=2,
            evidence=['FuelSystem'], evidence_card=[2],
            values=[[0.95, 0.50],
                    [0.05, 0.50]]
        )

        cpd_fuel_to_injector = TabularCPD(
            variable='fuel_to_injector', variable_card=2,
            evidence=['FuelSystem'], evidence_card=[2],
            values=[[0.90, 0.40],
                    [0.10, 0.60]] 
        )

        cpd_starts_and_stalls = TabularCPD(
            variable='starts_and_stalls', variable_card=2,
            evidence=['FuelSystem'], evidence_card=[2],
            values=[[0.8, 0.2], 
                    [0.2, 0.8]] 
        )

        cpd_engine_fires = TabularCPD(
            variable='engine_fires', variable_card=2,
            evidence=['FuelSystem', 'IgnitionSystem'], evidence_card=[2, 2],
            values=[[0.95, 0.25, 0.30, 0.10],  
                    [0.05, 0.75, 0.70, 0.90]] 
        )

        cpd_spark_to_plugs = TabularCPD(
            variable='spark_to_plugs', variable_card=2,
            evidence=['IgnitionSystem'], evidence_card=[2],
            values=[[0.90, 0.30], 
                    [0.10, 0.70]]
        )

        cpd_spark_from_coil = TabularCPD(
            variable='spark_from_coil', variable_card=2,
            evidence=['IgnitionSystem'], evidence_card=[2],
            values=[[0.85, 0.25],
                    [0.15, 0.75]] 
        )

        cpd_coil_primary_voltage = TabularCPD(
            variable='coil_primary_voltage', variable_card=2,
            evidence=['IgnitionSystem'], evidence_card=[2],
            values=[[0.80, 0.20],  
                    [0.20, 0.80]]  
        )

        cpd_mechanical_distributor = TabularCPD(
            variable='mechanical_distributor', variable_card=2,
            evidence=['IgnitionSystem'], evidence_card=[2],
            values=[[0.75, 0.15],  
                    [0.25, 0.85]]  
        )

        cpd_obd_codes = TabularCPD(
            variable='obd_codes', variable_card=2,
            evidence=['SensorSystem'], evidence_card=[2],
            values=[[0.95, 0.50],  
                    [0.05, 0.50]]  
        )

        cpd_stalls_on_key_release = TabularCPD(
            variable='stalls_on_key_release', variable_card=2,
            evidence=['SensorSystem'], evidence_card=[2],
            values=[[0.90, 0.40],
                    [0.10, 0.60]]
        )

        cpd_stalls_in_rain = TabularCPD(
            variable='stalls_in_rain', variable_card=2,
            evidence=['SensorSystem'], evidence_card=[2],
            values=[[0.85, 0.50],
                    [0.15, 0.50]]
        )

        cpd_stalls_when_warm = TabularCPD(
            variable='stalls_when_warm', variable_card=2,
            evidence=['SensorSystem'], evidence_card=[2],
            values=[[0.80, 0.55],
                    [0.20, 0.45]]
        )

        cpd_stalls_when_cold = TabularCPD(
            variable='stalls_when_cold', variable_card=2,
            evidence=['SensorSystem'], evidence_card=[2],
            values=[[0.80, 0.55],
                    [0.20, 0.45]]
        )

        model.add_cpds(cpd_starter, cpd_battery, cpd_fuel, cpd_ignition, cpd_sensor,
                       cpd_starter_cranks, cpd_starter_spins, cpd_battery_voltage,
                       cpd_cleaned_terminals, cpd_fuel_to_filter, cpd_fuel_to_injector,
                       cpd_starts_and_stalls, cpd_engine_fires, cpd_spark_to_plugs,
                       cpd_spark_from_coil, cpd_coil_primary_voltage, cpd_mechanical_distributor,
                       cpd_obd_codes, cpd_stalls_on_key_release, cpd_stalls_in_rain,
                       cpd_stalls_when_warm, cpd_stalls_when_cold)

        assert model.check_model()

        return model
    
    def transform_evidence(self, user_input):
        for value in user_input:
            if user_input[value] == True:
                user_input[value] = 1
            else:
                user_input[value] = 0
        
        return user_input

    def infer_problem(self, evidence_dict):

        systems = ['StarterSystem', 'BatterySystem', 'FuelSystem', 
                'IgnitionSystem', 'SensorSystem']
        
        probabilities = {}
        
        for system in systems:
            result = self.inference.query(variables=[system], evidence=evidence_dict)
            probabilities[system] = result.values[1]
        
        return probabilities

class StartProblem(KnowledgeEngine):
    pass

class StartDiagnostic(KnowledgeEngine):
    """Expert system for diagnosing starting problems in an engine."""

    def __init__(self):
        super().__init__()
        self.evidence_list = []
        self.next_question = None
        self.current_fact = None
        self.diagnostic_complete = False
        self.diagnostic_result = None
        self.diagnostic_message = None

    def get_next_question(self):
        return self.next_question

    def set_next_question(self, question, fact):
        self.next_question = question
        self.current_fact = fact

    def process_answer(self, answer):
        """Procesa la respuesta del usuario y actualiza el estado"""
        if self.current_fact:
            self.declare(Fact(**{self.current_fact: answer}))
            self.evidence_list.append((self.current_fact, answer == 'yes'))

    def generate_diagnostic(self, evidence_dict, message=""):
        inference_engine = StartingInference()
        probabilities = inference_engine.infer_problem(evidence_dict)
        most_probable_problem = max(probabilities, key=probabilities.get)
        
        self.diagnostic_complete = True
        self.diagnostic_result = {
            "most_probable_problem": most_probable_problem,
            "probabilities": probabilities,
            "diagnostic_message": message
        }
        self.next_question = None
        return self.diagnostic_result

    @DefFacts()
    def initial_fact(self):
        yield Fact(action='diagnose')

    @Rule(Fact(action='diagnose'),
          NOT(Fact(starter_cranks=W())))
    def start_diagnosis(self):
        self.set_next_question(
            "When you turn the key or press the start button, does the starter crank the engine? "
            "Answer 'yes' if you hear the engine turning over, or 'no' if it remains silent.", 
            "starter_cranks"
        )

    @Rule(Fact(action='diagnose'),
          Fact(starter_cranks="no"),
          NOT(Fact(starter_spins=W())))
    def starter_does_not_crank(self):
        self.set_next_question(
            "Does the starter motor spin but fail to engage with the engine? "
            "This would sound like a whirring noise.", 
            "starter_spins"
        )

    @Rule(Fact(action='diagnose'),
          Fact(starter_spins="yes"))
    def starter_does_not_spin(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "The starter solenoid might be stuck or not receiving power. "
            "Inspect the flywheel for any missing teeth that could prevent engagement."
        )

    @Rule(Fact(action='diagnose'),
          Fact(starter_spins="no"),
          NOT(Fact(battery_voltage=W())))
    def starter_spins_check_battery(self):
        self.set_next_question(
            "Does the battery voltage read above 12 volts? "
            "Use a multimeter to measure it, or check for indications of a weak battery (dim lights, etc.).", 
            "battery_voltage"
        )

    @Rule(Fact(action='diagnose'),
          Fact(battery_voltage="no"))
    def low_battery_voltage(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "The battery might be discharged. "
            "Try jump-starting the car or replace the battery if needed. "
            "Also, ensure the battery is charging correctly when the engine runs."
        )

    @Rule(Fact(action='diagnose'),
          Fact(battery_voltage="yes"),
          NOT(Fact(cleaned_terminals=W())))
    def clean_terminals(self):
        self.set_next_question(
            "Are the battery terminals and cable connections clean and free of corrosion? "
            "Check for white or green buildup that can interfere with electrical flow.", 
            "cleaned_terminals"
        )

    @Rule(Fact(action='diagnose'),
          Fact(cleaned_terminals="no"))
    def dirty_terminals(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "The battery terminals and ground connectors might be corroded. "
            "Clean them thoroughly to restore proper electrical contact."
        )
        
    @Rule(Fact(action='diagnose'),
          Fact(cleaned_terminals="yes"))
    def check_starter_function(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Test the starter by bypassing it directly in neutral or park. "
            "If the starter doesn't engage or crank the engine, it may need to be replaced."
        )

    @Rule(Fact(action='diagnose'),
          Fact(starter_cranks="yes"),
          NOT(Fact(engine_fires=W())))
    def engine_fires(self):
        self.set_next_question(
            "Does the engine attempt to fire or start after the starter cranks the engine? "
            "For example, do you hear the engine catching or struggling to start?", 
            "engine_fires"
        )

    @Rule(Fact(action='diagnose'),
          Fact(engine_fires="no"),
          NOT(Fact(spark_to_plugs=W())))
    def check_spark_to_plugs(self):
        self.set_next_question(
            "Is there a spark reaching the spark plugs? "
            "This can be checked using a spark tester or by observing the plugs.", 
            "spark_to_plugs"
        )

    @Rule(Fact(action='diagnose'),
          Fact(spark_to_plugs="no"),
          NOT(Fact(spark_from_coil=W())))
    def no_spark_to_plugs(self):
        self.set_next_question(
            "Is there a spark coming from the ignition coil? "
            "You can test this with an inline spark tester connected to the coil.", 
            "spark_from_coil"
        )

    @Rule(Fact(action='diagnose'),
          Fact(spark_to_plugs="yes"),
          NOT(Fact(fuel_to_filter=W())))
    def spark_to_plugs_present(self):
        self.set_next_question(
            "Is fuel reaching the fuel filter? "
            "Inspect the fuel line leading to the filter for any blockages or issues.", 
            "fuel_to_filter"
        )

    @Rule(Fact(action='diagnose'),
          Fact(fuel_to_filter="no"))
    def no_fuel_to_filter(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Inspect the fuel pump, fuel filter, and fuel lines for blockages, leaks, or malfunctions. "
            "A failed fuel pump or clogged filter could prevent fuel from reaching the engine."
        )

    @Rule(Fact(action='diagnose'),
          Fact(fuel_to_filter="yes"),
          NOT(Fact(fuel_to_injector=W())))
    def fuel_to_filter_present(self):
        self.set_next_question(
            "Is fuel reaching the fuel injector? "
            "This can be checked by inspecting the injector lines or using a pressure tester.", 
            "fuel_to_injector"
        )

    @Rule(Fact(action='diagnose'),
          Fact(fuel_to_injector="no"))
    def no_fuel_to_injector(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Try using starter spray in the carburetor, throttle body, or intake manifold. "
            "This may help identify whether the issue is fuel delivery-related."
        )

    @Rule(Fact(action='diagnose'),
          Fact(fuel_to_injector="yes"))
    def fuel_to_injector_present(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "For single-point injection systems, check the throttle body for clogs or malfunctions. "
            "For electronic multi-point injection systems, consider a specialized diagnostic."
        )
    
    @Rule(Fact(action='diagnose'), Fact(spark_from_coil='no'))
    def check_coil_primary_voltage(self):
        self.set_next_question(
            "Is there 12 volts or more at the coil's primary terminal? "
            "Use a multimeter to check the voltage while the ignition is on.", 
            "coil_primary_voltage"
        )

    @Rule(Fact(action='diagnose'), Fact(coil_primary_voltage='no'))
    def no_coil_voltage(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Inspect the ignition wiring for damage or disconnections. "
            "Check the voltage regulator to ensure it is providing the correct output."
        )

    @Rule(Fact(action='diagnose'), Fact(coil_primary_voltage='yes'))
    def coil_voltage_present(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Test the ignition coil for shorts or internal damage. "
            "Measure the resistance of the secondary output wire to ensure proper operation."
        )

    @Rule(Fact(action='diagnose'), Fact(spark_from_coil='yes'))
    def check_distributor(self):
        self.set_next_question(
            "Is the vehicle equipped with a mechanical distributor? "
            "Mechanical distributors have points and condensers.", 
            "mechanical_distributor"
        )

    Rule(Fact(action='diagnose'), Fact(mechanical_distributor='yes'))
    def mechanical_distributor(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Inspect the distributor points, condenser, rotor, and cap for signs of wear or damage. "
            "Replace any faulty components."
        )

    @Rule(Fact(action='diagnose'), Fact(mechanical_distributor='no'))
    def electronic_distributor(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Refer to the vehicle's service manual for specific diagnostic procedures "
            "related to the electronic distributor."
        )

    @Rule(Fact(action='diagnose'), Fact(engine_fires='yes'))
    def engine_stalls(self):
        self.set_next_question(
            "Does the engine start but then stall after a short period? "
            "This might indicate an issue with the fuel system or ignition timing.", 
            "starts_and_stalls"
        )

    @Rule(Fact(action='diagnose'), Fact(starts_and_stalls='no'))
    def no_obd_codes(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Inspect the ignition timing, fuel system, and battery. "
            "Ensure the fuel pressure is adequate, and verify that the ignition components are functioning properly."
        )

    @Rule(Fact(action='diagnose'), Fact(starts_and_stalls='yes'))
    def check_obd_code(self):
        self.set_next_question(
            "Are On-Board Diagnostics (OBD) or blink codes available for troubleshooting? "
            "This requires a diagnostic scanner or observing flashing lights on the dash.", 
            "obd_codes"
        )

    @Rule(Fact(action='diagnose'), Fact(obd_codes='no'))
    def interpret_obd_codes(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Use an OBD reader or interpret the blink codes to pinpoint the issue. "
            "These codes can guide you to the specific malfunction."
        )

    @Rule(Fact(action='diagnose'), Fact(obd_codes='yes'))
    def check_stall_conditions(self):
        self.set_next_question(
            "Does the engine stall when you release the key after starting? "
            "This could indicate an issue with the ignition switch or related wiring.", 
            "stalls_on_key_release"
        )

    @Rule(Fact(action='diagnose'), Fact(stalls_on_key_release='yes'))
    def stalls_on_key_release(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Inspect the ignition circuit and key switch for faults. "
            "This may include worn-out contacts, loose connections, or a faulty switch mechanism."
        )

    @Rule(Fact(action='diagnose'), Fact(stalls_on_key_release='no'))
    def check_weather_conditions(self):
        self.set_next_question(
            "Does the vehicle stall during rainy or wet conditions? "
            "This could indicate issues with moisture affecting electrical components.", 
            "stalls_in_rain"
        )

    @Rule(Fact(action='diagnose'), Fact(stalls_in_rain='yes'))
    def stalls_in_rain(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Inspect the ignition coils and distributor for cracks or moisture. "
            "Look for visible electrical arcs that may indicate short circuits. Dry and reseal if needed."
        )

    @Rule(Fact(action='diagnose'), Fact(stalls_in_rain='no'))
    def check_temperature_conditions(self):
        self.set_next_question(
            "Does the vehicle stall when the engine is warm or after running for a while? "
            "This could indicate issues with the fuel system or idle settings.", 
            "stalls_when_warm"
        )

    @Rule(Fact(action='diagnose'), Fact(stalls_when_warm='yes'))
    def stalls_when_warm(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Adjust the idle speed, clean the fuel filters, and inspect for vacuum leaks. "
            "Warm stalling could also indicate issues with the throttle body or a failing fuel pump."
        )

    @Rule(Fact(action='diagnose'), Fact(stalls_when_warm='no'))
    def stalls_when_cold(self):
        self.set_next_question(
            "Does the vehicle stall when the engine is cold or during cold starts? "
            "This could be related to the choke system or air-fuel mixture.", 
            "stalls_when_cold"
        )

    @Rule(Fact(action='diagnose'), Fact(stalls_when_cold='yes'))
    def stalls_when_cold(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Inspect the choke system and Exhaust Gas Recirculation (EGR) system for proper operation. "
            "Check for vacuum leaks and ensure that the air intake system is free of obstructions."
        )