from experta import *
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import logging

# Suppress the logging output from Experta
logging.getLogger("experta.watchers").setLevel(logging.ERROR)


class StartingInference:
    def __init__(self):
        self.model = self._build_model()
        self.inference = VariableElimination(self.model)

    def _build_model(self):
        # Create the Bayesian Network structure
        model = BayesianNetwork([
            ('BrakeEffectiveness', 'brakes_stop_car'),
            ('BrakeEffectiveness', 'pedal_to_floor'),
            ('BrakeEffectiveness', 'brake_fluid_ok'),
            ('BrakeEffectiveness', 'brake_light'),
            ('ParkingBrake', 'parking_brake_failure'),
            ('ParkingBrake', 'rear_wheel_locked'),
            ('ParkingBrake', 'ratchets_without_force'),
            ('WheelResistance', 'wheel_drag_much'),
            ('WheelResistance', 'need_pump_brakes'),
            ('WheelResistance', 'only_after_turning'),
            ('BrakePadOrRotorIssue', 'making_noise'),
            ('BrakePadOrRotorIssue', 'squealing'),
            ('BrakePadOrRotorIssue', 'clunks'),
            ('BrakePadOrRotorIssue', 'scrape_or_grind'),
            ('BrakePadOrRotorIssue', 'rattles'),
            ('BrakeBehavior', 'brakes_pull'),
            ('BrakeBehavior', 'jerky_pulsing')
        ])

        # Root cause probabilities
        cpd_BrakeEffectiveness = TabularCPD('BrakeEffectiveness', 2, [[0.8], [0.2]])
        cpd_ParkingBrake = TabularCPD('ParkingBrake', 2, [[0.9], [0.1]])
        cpd_WheelResistance = TabularCPD('WheelResistance', 2, [[0.85], [0.15]])
        cpd_BrakePadOrRotorIssue = TabularCPD('BrakePadOrRotorIssue', 2, [[0.75], [0.25]])
        cpd_BrakeBehavior = TabularCPD('BrakeBehavior', 2, [[0.9], [0.1]])

        # Conditional probabilities for symptoms
        cpd_brakes_stop_car = TabularCPD('brakes_stop_car', 2,
                                         [[0.9, 0.1], [0.1, 0.9]],
                                         evidence=['BrakeEffectiveness'], evidence_card=[2])

        cpd_pedal_to_floor = TabularCPD('pedal_to_floor', 2,
                                        [[0.8, 0.2], [0.2, 0.8]],
                                        evidence=['BrakeEffectiveness'], evidence_card=[2])

        cpd_brake_fluid_ok = TabularCPD('brake_fluid_ok', 2,
                                        [[0.7, 0.3], [0.3, 0.7]],
                                        evidence=['BrakeEffectiveness'], evidence_card=[2])

        cpd_brake_light = TabularCPD('brake_light', 2,
                                     [[0.6, 0.4], [0.4, 0.6]],
                                     evidence=['BrakeEffectiveness'], evidence_card=[2])

        cpd_parking_brake_failure = TabularCPD('parking_brake_failure', 2,
                                               [[0.85, 0.15], [0.15, 0.85]],
                                               evidence=['ParkingBrake'], evidence_card=[2])

        cpd_rear_wheel_locked = TabularCPD('rear_wheel_locked', 2,
                                           [[0.7, 0.3], [0.3, 0.7]],
                                           evidence=['ParkingBrake'], evidence_card=[2])

        cpd_ratchets_without_force = TabularCPD('ratchets_without_force', 2,
                                                [[0.8, 0.2], [0.2, 0.8]],
                                                evidence=['ParkingBrake'], evidence_card=[2])

        cpd_wheel_drag_much = TabularCPD('wheel_drag_much', 2,
                                         [[0.9, 0.1], [0.1, 0.9]],
                                         evidence=['WheelResistance'], evidence_card=[2])

        cpd_need_pump_brakes = TabularCPD('need_pump_brakes', 2,
                                          [[0.7, 0.3], [0.3, 0.7]],
                                          evidence=['WheelResistance'], evidence_card=[2])

        cpd_only_after_turning = TabularCPD('only_after_turning', 2,
                                            [[0.6, 0.4], [0.4, 0.6]],
                                            evidence=['WheelResistance'], evidence_card=[2])

        cpd_making_noise = TabularCPD('making_noise', 2,
                                      [[0.9, 0.1], [0.1, 0.9]],
                                      evidence=['BrakePadOrRotorIssue'], evidence_card=[2])

        cpd_squealing = TabularCPD('squealing', 2,
                                   [[0.8, 0.2], [0.2, 0.8]],
                                   evidence=['BrakePadOrRotorIssue'], evidence_card=[2])

        cpd_clunks = TabularCPD('clunks', 2,
                                [[0.7, 0.3], [0.3, 0.7]],
                                evidence=['BrakePadOrRotorIssue'], evidence_card=[2])

        cpd_scrape_or_grind = TabularCPD('scrape_or_grind', 2,
                                         [[0.6, 0.4], [0.4, 0.6]],
                                         evidence=['BrakePadOrRotorIssue'], evidence_card=[2])

        cpd_rattles = TabularCPD('rattles', 2,
                                 [[0.7, 0.3], [0.3, 0.7]],
                                 evidence=['BrakePadOrRotorIssue'], evidence_card=[2])

        cpd_brakes_pull = TabularCPD('brakes_pull', 2,
                                     [[0.8, 0.2], [0.2, 0.8]],
                                     evidence=['BrakeBehavior'], evidence_card=[2])

        cpd_jerky_pulsing = TabularCPD('jerky_pulsing', 2,
                                       [[0.7, 0.3], [0.3, 0.7]],
                                       evidence=['BrakeBehavior'], evidence_card=[2])

        # Add CPDs to the model
        model.add_cpds(
            cpd_BrakeEffectiveness, cpd_ParkingBrake, cpd_WheelResistance,
            cpd_BrakePadOrRotorIssue, cpd_BrakeBehavior, cpd_brakes_stop_car,
            cpd_pedal_to_floor, cpd_brake_fluid_ok, cpd_brake_light, cpd_parking_brake_failure,
            cpd_rear_wheel_locked, cpd_ratchets_without_force, cpd_wheel_drag_much,
            cpd_need_pump_brakes, cpd_only_after_turning, cpd_making_noise,
            cpd_squealing, cpd_clunks, cpd_scrape_or_grind, cpd_rattles,
            cpd_brakes_pull, cpd_jerky_pulsing
        )

        # Validate the model
        assert model.check_model()

        return model

    def infer_problem(self, evidence_dict):
        """
        Infers probabilities of general problems given observed evidence.

        Args:
            evidence_dict: Dictionary with observed evidence.

        Returns:
            dict: Probabilities of each general problem.
        """
        problems = ['BrakeEffectiveness', 'ParkingBrake', 'WheelResistance',
                    'BrakePadOrRotorIssue', 'BrakeBehavior']

        probabilities = {}

        for problem in problems:
            result = self.inference.query(variables=[problem], evidence=evidence_dict)
            probabilities[problem] = result.values[1]

        problem_mapping = {
            'BrakeEffectiveness': 'Issues with braking effectiveness',
            'ParkingBrake': 'Parking brake issues',
            'WheelResistance': 'Wheel resistance issues',
            'BrakePadOrRotorIssue': 'Brake pad or rotor issues',
            'BrakeBehavior': 'Braking behavior issues'
        }

        mapped_probabilities = {problem_mapping[problem]: prob for problem, prob in probabilities.items()}

        return mapped_probabilities


class BrakeProblem(Fact):
    """Fact representation for brake problems."""
    pass

class BrakeDiagnostic(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.evidence_list = []
        self.next_question = None
        self.current_fact = None
        self.diagnostic_complete = False
        self.diagnostic_result = None
        self.diagnostic_message = None

    @DefFacts()
    def initial_fact(self):
        yield Fact(action="diagnose_brakes")

    def get_next_question(self):
        return self.next_question

    def set_next_question(self, question, fact):
        self.next_question = question
        self.current_fact = fact

    def process_answer(self, answer):
        """Procesa la respuesta del usuario y actualiza el estado"""
        if self.current_fact:
            self.declare(BrakeProblem(**{self.current_fact: answer}))
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

    # Initial question
    @Rule(Fact(action="diagnose_brakes"),
          NOT(BrakeProblem(brakes_stop_car=W())))
    def ask_brakes_stop_car(self):
        self.set_next_question("Do the brakes stop the car?", "brakes_stop_car")

    # Branch: Brakes stop car = NO
    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(brakes_stop_car="no"),
          NOT(BrakeProblem(pedal_to_floor=W())))
    def ask_pedal_to_floor(self):
        self.set_next_question("Does the pedal go to floor?", "pedal_to_floor")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(brakes_stop_car="no"),
          BrakeProblem(pedal_to_floor="yes"),
          NOT(BrakeProblem(brake_fluid_ok=W())))
    def ask_brake_fluid(self):
        self.set_next_question("Is brake fluid level OK?", "brake_fluid_ok")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(brake_fluid_ok="no"))
    def low_brake_fluid(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Fill to line. If brakes are soft, bleed lines following service manual."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(brake_fluid_ok="yes"),
          NOT(BrakeProblem(brake_light=W())))
    def ask_brake_light(self):
        self.set_next_question("Is the brake warning light on?", "brake_light")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(brake_fluid_ok="yes"),
          BrakeProblem(brake_light="no"))
    def check_service_manual(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Likely power assist related, see service manual."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(brake_fluid_ok="yes"),
          BrakeProblem(brake_light="yes"))
    def check_power_booster(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "If parking brake released see service manual for power booster problem or anti-lock failure."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(pedal_to_floor="no"))
    def check_linkage(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Pedal linkage, glazed, frozen calipers, pinched lines, or booster failure."
        )

    # Branch: Brakes stop car = YES
    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(brakes_stop_car="yes"),
          NOT(BrakeProblem(parking_brake_failure=W())))
    def ask_parking_brake(self):
        self.set_next_question("Is there a parking brake failure?", "parking_brake_failure")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(parking_brake_failure="yes"),
          NOT(BrakeProblem(rear_wheel_locked=W())))
    def ask_rear_wheel_locked(self):
        self.set_next_question("Is rear wheel locked?", "rear_wheel_locked")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(rear_wheel_locked="yes"))
    def spring_return_failure(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Spring return failure or cable rusted bound."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(rear_wheel_locked="no"),
          NOT(BrakeProblem(ratchets_without_force=W())))
    def ask_ratchets(self):
        self.set_next_question("Does it ratchet without force?", "ratchets_without_force")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(ratchets_without_force="yes"))
    def cable_problem(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Cable stretched or broken, freeze adjuster."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(ratchets_without_force="no"))
    def cable_problem1(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Shoes worn out, glazed, fluid in drums."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(parking_brake_failure="no"),
          NOT(BrakeProblem(wheel_drag_much=W())))
    def ask_wheel_drag(self):
        self.set_next_question("Do the wheels drag too much?", "wheel_drag_much")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(wheel_drag_much="yes"))
    def diagnostic_drag(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Stuck piston, hydraulic lock, over adjusted drum shoes, warped rotor."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(wheel_drag_much="no"),
          NOT(BrakeProblem(need_pump_brakes=W())))
    def ask_need_pump(self):
        self.set_next_question("Need to pump up brakes?", "need_pump_brakes")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(need_pump_brakes="yes"),
          NOT(BrakeProblem(only_after_turning=W())))
    def ask_after_turning(self):
        self.set_next_question("Only after turning?", "only_after_turning")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(need_pump_brakes="yes"),
          BrakeProblem(only_after_turning="yes"))
    def wheel_bearing_problem(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Front wheel bearings worn; axle loose; wheel lugs loose."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(need_pump_brakes="yes"),
          BrakeProblem(only_after_turning="no"))
    def air_system_problem(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Air in system; fluid leak."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(need_pump_brakes="no"),
          NOT(BrakeProblem(making_noise=W())))
    def ask_making_noise(self):
        self.set_next_question("Making noise?", "making_noise")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          NOT(BrakeProblem(squealing=W())))
    def ask_squealing(self):
        self.set_next_question("Squealing?", "squealing")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          BrakeProblem(squealing="yes"))
    def check_pads_wear(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Check pads and shoes for wear, foreign objects."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          BrakeProblem(squealing="no"),
          NOT(BrakeProblem(clunks=W())))
    def ask_clunks(self):
        self.set_next_question("Clunks?", "clunks")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          BrakeProblem(squealing="no"),
          BrakeProblem(clunks="yes"))
    def caliper_bolt(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Caliper bolt loose, suspension problem (see clicking noises diagnostic)."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          BrakeProblem(squealing="no"),
          BrakeProblem(clunks="no"),
          NOT(BrakeProblem(scrape_or_grind=W())))
    def ask_scrape_grind(self):
        self.set_next_question("Scrape or grind?", "scrape_or_grind")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          BrakeProblem(squealing="no"),
          BrakeProblem(clunks="no"),
          BrakeProblem(scrape_or_grind="yes"))
    def brake_pad_worn(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Broken pad or shoe (facing, warning sound) or excessive wear."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          BrakeProblem(squealing="no"),
          BrakeProblem(clunks="no"),
          BrakeProblem(scrape_or_grind="no"),
          NOT(BrakeProblem(rattles=W())))
    def ask_rattles(self):
        self.set_next_question("Rattles?", "rattles")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          BrakeProblem(squealing="no"),
          BrakeProblem(clunks="no"),
          BrakeProblem(scrape_or_grind="no"),
          BrakeProblem(rattles="yes"))
    def anti_rattle_problem(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Anti-rattle clips on disc pads missing or installed wrong."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="yes"),
          BrakeProblem(squealing="no"),
          BrakeProblem(clunks="no"),
          BrakeProblem(scrape_or_grind="no"),
          BrakeProblem(rattles="no"))
    def rotor_warped(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Chirps and ticks that increase with speed due to rotor warp or run out."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="no"),
          NOT(BrakeProblem(brakes_pull=W())))
    def ask_brakes_pull(self):
        self.set_next_question("Do brakes pull?", "brakes_pull")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="no"),
          BrakeProblem(brakes_pull="yes"))
    def front_brake_issue(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Front brake issue - stuck or cocked piston, air or crimp in line, master cylinder problem."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="no"),
          BrakeProblem(brakes_pull="no"),
          NOT(BrakeProblem(jerky_pulsing=W())))
    def ask_jerky_pulsing(self):
        self.set_next_question("Jerky pulsing?", "jerky_pulsing")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="no"),
          BrakeProblem(brakes_pull="no"),
          BrakeProblem(jerky_pulsing="yes"))
    def antilock_issue(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Anti-lock brake issue, deformed drum or rotor (test with parking brake)."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="no"),
          BrakeProblem(brakes_pull="no"),
          BrakeProblem(jerky_pulsing="no"),
          NOT(BrakeProblem(hard_braking=W())))
    def ask_hard_braking(self):
        self.set_next_question("Hard braking?", "hard_braking")

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="no"),
          BrakeProblem(brakes_pull="no"),
          BrakeProblem(jerky_pulsing="no"),
          BrakeProblem(hard_braking="yes"))
    def power_boost_issue(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "Worn pads, shoes, bound piston, power boost problem."
        )

    @Rule(Fact(action="diagnose_brakes"),
          BrakeProblem(making_noise="no"),
          BrakeProblem(brakes_pull="no"),
          BrakeProblem(jerky_pulsing="no"),
          BrakeProblem(hard_braking="no"))
    def warning_light(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "If brake warning light on and parking brake is released, see service manual for codes."
        )
