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
            ('Suspension_issues', 'clunk_or_single_tick'),
            ('Suspension_issues', 'noise_on_bumps'),
            
            ('Brake_and_wheel_problems', 'clunk_or_single_tick'),
            ('Brake_and_wheel_problems', 'ticks_when_moving'),
            ('Brake_and_wheel_problems', 'ticks_in_reverse'),
            ('Brake_and_wheel_problems', 'ticks_slow_speed'),
            ('Brake_and_wheel_problems', 'changed_tires'),
            ('Brake_and_wheel_problems', 'removed_hubcaps'),
            ('Brake_and_wheel_problems', 'inspect_treads'),
            
            ('Transmission_or_drivetrain', 'ticks_when_moving'),
            ('Transmission_or_drivetrain', 'ticks_in_neutral'),
            ('Transmission_or_drivetrain', 'frequency_changes'),
            
            ('Exhaust_or_engine_noises', 'ticks_when_cold'),
            ('Exhaust_or_engine_noises', 'windshield_wipers_radio'),
            
            ('CV_joint_or_alignment', 'ticks_in_turns'),
            ('CV_joint_or_alignment', 'ticks_when_moving')
        ])

        cpd_suspension = TabularCPD(
            variable='Suspension_issues',
            variable_card=2,
            values=[[0.85], [0.15]]
        )

        cpd_brake_wheel = TabularCPD(
            variable='Brake_and_wheel_problems',
            variable_card=2,
            values=[[0.80], [0.20]]
        )

        cpd_transmission = TabularCPD(
            variable='Transmission_or_drivetrain',
            variable_card=2,
            values=[[0.90], [0.10]]
        )

        cpd_exhaust = TabularCPD(
            variable='Exhaust_or_engine_noises',
            variable_card=2,
            values=[[0.85], [0.15]]
        )

        cpd_cv_joint = TabularCPD(
            variable='CV_joint_or_alignment',
            variable_card=2,
            values=[[0.88], [0.12]]
        )

        cpd_clunk = TabularCPD(
            variable='clunk_or_single_tick',
            variable_card=2,
            evidence=['Suspension_issues', 'Brake_and_wheel_problems'],
            evidence_card=[2, 2],
            values=[
                [0.95, 0.40, 0.30, 0.05],  
                [0.05, 0.60, 0.70, 0.95]   
            ]
        )

        cpd_bumps = TabularCPD(
            variable='noise_on_bumps',
            variable_card=2,
            evidence=['Suspension_issues'],
            evidence_card=[2],
            values=[
                [0.90, 0.15],  
                [0.10, 0.85]   
            ]
        )

        cpd_moving = TabularCPD(
            variable='ticks_when_moving',
            variable_card=2,
            evidence=['Brake_and_wheel_problems', 'Transmission_or_drivetrain', 'CV_joint_or_alignment'],
            evidence_card=[2, 2, 2],
            values=[
                [0.95, 0.30, 0.20, 0.10, 0.30, 0.15, 0.05, 0.01], 
                [0.05, 0.70, 0.80, 0.90, 0.70, 0.85, 0.95, 0.99]   
            ]
        )

        cpd_neutral = TabularCPD(
            variable='ticks_in_neutral',
            variable_card=2,
            evidence=['Transmission_or_drivetrain'],
            evidence_card=[2],
            values=[
                [0.90, 0.20],  
                [0.10, 0.80]  
            ]
        )

        cpd_reverse = TabularCPD(
            variable='ticks_in_reverse',
            variable_card=2,
            evidence=['Brake_and_wheel_problems'],
            evidence_card=[2],
            values=[
                [0.95, 0.30],  
                [0.05, 0.70]  
            ]
        )

        cpd_frequency = TabularCPD(
            variable='frequency_changes',
            variable_card=2,
            evidence=['Transmission_or_drivetrain'],
            evidence_card=[2],
            values=[
                [0.85, 0.25], 
                [0.15, 0.75]   
            ]
        )

        cpd_cold = TabularCPD(
            variable='ticks_when_cold',
            variable_card=2,
            evidence=['Exhaust_or_engine_noises'],
            evidence_card=[2],
            values=[
                [0.90, 0.20],  
                [0.10, 0.80]   
            ]
        )

        cpd_wipers = TabularCPD(
            variable='windshield_wipers_radio',
            variable_card=2,
            evidence=['Exhaust_or_engine_noises'],
            evidence_card=[2],
            values=[
                [0.80, 0.30],  
                [0.20, 0.70]   
            ]
        )

        cpd_turns = TabularCPD(
            variable='ticks_in_turns',
            variable_card=2,
            evidence=['CV_joint_or_alignment'],
            evidence_card=[2],
            values=[
                [0.95, 0.15],  
                [0.05, 0.85]   
            ]
        )

        cpd_tires = TabularCPD(
            variable='changed_tires',
            variable_card=2,
            evidence=['Brake_and_wheel_problems'],
            evidence_card=[2],
            values=[
                [0.90, 0.60],  
                [0.10, 0.40]   
            ]
        )

        cpd_hubcaps = TabularCPD(
            variable='removed_hubcaps',
            variable_card=2,
            evidence=['Brake_and_wheel_problems'],
            evidence_card=[2],
            values=[
                [0.95, 0.70],  
                [0.05, 0.30]   
            ]
        )

        cpd_treads = TabularCPD(
            variable='inspect_treads',
            variable_card=2,
            evidence=['Brake_and_wheel_problems'],
            evidence_card=[2],
            values=[
                [0.80, 0.30],  
                [0.20, 0.70]   
            ]
        )

        cpd_slow = TabularCPD(
            variable='ticks_slow_speed',
            variable_card=2,
            evidence=['Brake_and_wheel_problems'],
            evidence_card=[2],
            values=[
                [0.90, 0.20],  
                [0.10, 0.80]   
            ]
        )

        model.add_cpds(
            cpd_suspension, cpd_brake_wheel, cpd_transmission, cpd_exhaust, cpd_cv_joint,
            cpd_clunk, cpd_bumps, cpd_moving, cpd_neutral, cpd_reverse, cpd_frequency,
            cpd_cold, cpd_wipers, cpd_turns, cpd_tires, cpd_hubcaps, cpd_treads, cpd_slow
        )
        
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

        systems =  [
            'Suspension_issues',
            'Brake_and_wheel_problems',
            'Transmission_or_drivetrain',
            'Exhaust_or_engine_noises',
            'CV_joint_or_alignment'
        ] 
        
        probabilities = {}
        
        for system in systems:
            result = self.inference.query(variables=[system], evidence=evidence_dict)
            probabilities[system] = result.values[1]
        
        return probabilities

class SoundProblem(Fact):
    """Facts about the car sound"""
    pass

class SoundDiagnostic(KnowledgeEngine):
    """Expert system for diagnosing strange sounds in a vehicle."""

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
            self.declare(SoundProblem(**{self.current_fact: answer}))
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
        yield Fact(action="diagnose_sound")

    @Rule(Fact(action="diagnose_sound"),
          NOT(SoundProblem(clunk_or_single_tick=W())))
    def ask_clunk_or_single_tick(self):
        self.set_next_question(
            "Can you describe the noise? Does it sound like a loud clunk or a single ticking noise? "
            "This information will help us narrow down the potential issue.", 
            "clunk_or_single_tick"
        )

    @Rule(Fact(action="diagnose_sound"),
          SoundProblem(clunk_or_single_tick="yes"),
          NOT(SoundProblem(noise_on_bumps=W())))
    def ask_noise_on_bumps(self):
        self.set_next_question(
            "Have you noticed if the noise occurs only when driving over bumps or uneven surfaces? "
            "For example, does it happen when the car experiences vertical motion?", 
            "noise_on_bumps"
        )

    @Rule(Fact(action="diagnose_sound"),
          SoundProblem(clunk_or_single_tick="yes"),
          SoundProblem(noise_on_bumps="yes"))
    def check_suspension(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "The noise might be related to the suspension system. Inspect the struts, shocks, springs, "
            "and frame welds for damage or wear."
        )

    @Rule(Fact(action="diagnose_sound"),
          SoundProblem(clunk_or_single_tick="yes"),
          SoundProblem(noise_on_bumps="no"))
    def check_components(self):
        self.generate_diagnostic(
            dict(self.evidence_list),
            "The noise might involve components like ball joints, brakes (pads and rotors), "
            "rack and pinion, tie rod ends, or motor mounts. Check these parts carefully."
        )

    @Rule(Fact(action="diagnose_sound"),
          SoundProblem(clunk_or_single_tick="no"),
          NOT(SoundProblem(ticks_when_moving=W())))
    def ask_ticks_when_moving(self):
        self.set_next_question(
            "Does the noise occur only when the vehicle is in motion, or does it also happen when stationary? "
            "This distinction will help pinpoint the source.", 
            "ticks_when_moving"
        )

    #@Rule(Fact(action="diagnose_sound"),
    #      SoundProblem(clunk_or_single_tick="no"),
    #      SoundProblem(ticks_when_moving="no"),
    #      NOT(SoundProblem(time_bomb=W())))
    #def ask_time_bomb(self):
    #    self.set_next_question(
    #        "Does it sound like a ticking noise coming from under the seat, possibly resembling a time bomb? "
    #        "This could indicate an unusual issue.", 
    #        "time_bomb"
    #    )
    
    #@Rule(Fact(action="diagnose_sound"), 
    #  SoundProblem(clunk_or_single_tick="no"), 
    #  SoundProblem(ticks_when_moving="no"), 
    #  SoundProblem(time_bomb="yes"))
    #def google_bomb(self):
    #    self.generate_diagnostic(
    #        dict(self.evidence_list), 
    #        "This might not be a typical car problem. Please consult online resources or experts for "
    #        "guidance on handling unusual ticking noises like this."
    #    )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        NOT(SoundProblem(ticks_in_neutral=W())))
    def ask_ticks_neutral(self):
        self.set_next_question(
            "Does the ticking noise occur when the car is rolling in neutral? "
            "This will help us determine if the issue is related to the drivetrain or the engine. (yes/no)",
            "ticks_in_neutral"
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="no"), 
        NOT(SoundProblem(ticks_in_reverse=W())))
    def ask_ticks_reverse(self):
        self.set_next_question(
            "Does the ticking noise occur only when the car is in reverse? "
            "This could indicate a problem with the transmission or related components. (yes/no)",
            "ticks_in_reverse"
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="no"), 
        SoundProblem(ticks_in_reverse="yes"))
    def check_brake_adjuster(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "The issue might be caused by a rear brake adjuster. Ensure the parking brake is fully released, "
            "and inspect for any signs of improper adjustment."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="no"), 
        SoundProblem(ticks_in_reverse="no"))
    def ask_wheel_rotation(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "The noise might be coming from the transmission. Check the transmission fluid and filter for "
            "any irregularities or contamination."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        NOT(SoundProblem(frequency_changes=W())))
    def ask_frequency_changes(self):
        self.set_next_question(
            "Does the frequency of the ticking noise decrease or change when shifting gears? "
            "This will help identify if the issue is related to the transmission or engine timing. (yes/no)",
            "frequency_changes"
        )

    #@Rule(Fact(action="diagnose_sound"), 
    #    SoundProblem(clunk_or_single_tick="no"), 
    #    OR(SoundProblem(time_bomb="no"), SoundProblem(frequency_changes="yes")), 
    #    NOT(SoundProblem(ticks_when_cold=W())))
    #def ask_ticks_when_cold(self):
    #    print(
    #        "Try to pinpoint the location of the ticking noise using a hearing tube or a long screwdriver. "
    #        "Place the handle near your ear to amplify the sound source."
    #    )
    #    self.set_next_question(
    #        "Does the ticking noise only happen when the engine is cold? (yes/no)",
    #        "ticks_when_cold"
    #    )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_cold="yes"))
    def check_exhaust(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "The noise might be caused by an exhaust system issue. Inspect the exhaust pipe near the catalytic "
            "converter for leaks and listen for any rattling noises near the valve cover."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_cold="no"), 
        NOT(SoundProblem(windshield_wipers_radio=W())))
    def ask_windshield_wipers_radio(self):
        self.set_next_question(
            "Are the windshield wipers and radio turned off while you're hearing the noise? "
            "This helps rule out external distractions causing the sound. (yes/no)",
            "windshield_wipers_radio"
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_cold="no"), 
        SoundProblem(windshield_wipers_radio="no"))
    def check_silly_stuff(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "Always check for unusual causes, such as passengers tapping on the roof or random objects causing noise."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_cold="no"), 
        SoundProblem(windshield_wipers_radio="yes"))
    def final_checks(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "Inspect for pulley wobble or belt wear. Check for exhaust manifold leaks. "
            "If the sound persists, get someone with better hearing to help localize the source on the engine."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(frequency_changes="no"), 
        NOT(SoundProblem(ticks_in_turns=W())))
    def ask_ticks_in_turns(self):
        self.set_next_question(
            "Does the ticking noise occur only when taking turns or sharp curves? "
            "This could indicate an issue with the CV joint or related components. (yes/no)",
            "ticks_in_turns"
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_turns="yes"))
    def check_cv_joint(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "The CV joint might be failing. Oversized tires rubbing against the wheel well could also be the cause."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        NOT(SoundProblem(changed_tires=W())))
    def ask_changed_tires(self):
        self.set_next_question(
            "Have you recently changed the tires? "
            "This could help identify whether the sound is related to improper tire installation. (yes/no)",
            "changed_tires"
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        SoundProblem(changed_tires="yes"))
    def check_wheel_lugs(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "Stop driving immediately! Ensure that the wheel lugs are tightened properly."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        SoundProblem(changed_tires="no"), 
        NOT(SoundProblem(removed_hubcaps=W())))
    def ask_removed_hubcaps(self):
        self.set_next_question(
            "Have you removed the hubcaps recently? "
            "This helps identify if the sound is related to loose or misaligned hubcaps. (yes/no)",
            "removed_hubcaps"
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="no"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        SoundProblem(changed_tires="no"), 
        SoundProblem(removed_hubcaps="no"))
    def remove_hubcaps_check(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "Before proceeding, remove the hubcaps. Loose wire retainers or trapped pebbles may be causing the ticking noise."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        SoundProblem(changed_tires="no"), 
        SoundProblem(removed_hubcaps="yes"), 
        NOT(SoundProblem(inspect_treads=W())))
    def ask_inspect_treads(self):
        self.set_next_question(
            "Have you inspected the tire treads for embedded objects like nails or stones? "
            "This could explain the noise. (yes/no)",
            "inspect_treads"
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(changed_tires="no"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        SoundProblem(removed_hubcaps="yes"), 
        SoundProblem(inspect_treads="no"))
    def check_nails_stones(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "Inspect the tire treads for nails, stones, or other debris embedded in the rubber."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(changed_tires="no"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        SoundProblem(removed_hubcaps="yes"), 
        SoundProblem(inspect_treads="yes"), 
        NOT(SoundProblem(ticks_slow_speed=W())))
    def ask_ticks_slow_speed(self):
        self.set_next_question(
            "Does the ticking noise occur only at slow speeds? "
            "This helps narrow down potential issues with the wheels or axles. (yes/no)",
            "ticks_slow_speed"
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(changed_tires="no"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        SoundProblem(removed_hubcaps="yes"), 
        SoundProblem(inspect_treads="yes"), 
        SoundProblem(ticks_slow_speed="yes"))
    def check_wheel_covers(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "Check the bolted wheel covers or hub protectors for loose parts or pebbles trapped inside."
        )

    @Rule(Fact(action="diagnose_sound"), 
        SoundProblem(clunk_or_single_tick="no"), 
        SoundProblem(ticks_when_moving="yes"), 
        SoundProblem(ticks_in_neutral="yes"), 
        SoundProblem(changed_tires="no"), 
        SoundProblem(frequency_changes="no"), 
        SoundProblem(ticks_in_turns="no"), 
        SoundProblem(removed_hubcaps="yes"), 
        SoundProblem(inspect_treads="yes"), 
        SoundProblem(ticks_slow_speed="no"))
    def check_brake_pads(self):
        self.generate_diagnostic(
            dict(self.evidence_list), 
            "The noise could be caused by brake pads ticking on a warped rotor. Also, inspect the axles for signs of rubbing."
        )