from fpdf import FPDF
import os
from configparser import ConfigParser

config=ConfigParser
# Create a directory for the files
output_dir = config["POLICY_MANUALS"]["directory"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class BreezeLitePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'BREEZELITE GLOBAL - OFFICIAL OPERATING MANUAL', 0, 1, 'C')
        self.ln(5)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 7, f"{title}", 0, 1, 'L', 1)
        self.ln(3)

    def section_body(self, text):
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 5, text)
        self.ln(4)

# 10 Different Models with specific exclusions and realistic operating steps
models = [
    {
        "name": "BreezeLite BLD-150 Everyday", "power": "1875W DC Motor", 
        "tech": "Basic Ionic Smoothing", 
        "ops": "1. Plug into a 125V AC outlet.\n2. Select Low speed for styling or High speed for drying.\n3. Always keep the dryer 6-8 inches from hair.",
        "extra_exclusion": "- Damage to the plastic intake housing due to high-impact drops."
    },
    {
        "name": "BreezeLite BLD-2000 Pro-Salon", "power": "2200W Professional AC Motor", 
        "tech": "Ceramic-Tourmaline Grille", 
        "ops": "1. Turn on unit using the rocker switches.\n2. Use the Cool Shot button to set the style after drying.\n3. Designed for extended use in professional environments.",
        "extra_exclusion": "- Buildup of professional styling products (hairspray/gels) inside the motor housing."
    },
    {
        "name": "BreezeLite BLD-X1 Infrared", "power": "1600W Low-EMF Motor", 
        "tech": "Infrared Light Therapy", 
        "ops": "1. Ensure the infrared lamp is active during drying.\n2. Move the dryer constantly to prevent localized heat spots.\n3. Use the concentrator for precision smoothing.",
        "extra_exclusion": "- Cracking or failure of the Infrared emitter bulb due to external pressure."
    },
    {
        "name": "BreezeLite BLD-Turbo 2400", "power": "2400W High-Velocity Motor", 
        "tech": "Aero-Flow Turbine Tech", 
        "ops": "1. Engage Turbo mode only after rough-drying hair on standard speed.\n2. Maintain airflow by keeping the rear intake away from loose clothing.",
        "extra_exclusion": "- Damage to the turbine fan blades caused by foreign objects entering the air intake."
    },
    {
        "name": "BreezeLite BLD-Sonic 5", "power": "1600W Brushless Digital Motor", 
        "tech": "Microprocessor Heat Control", 
        "ops": "1. Power on and wait 2 seconds for sensor calibration.\n2. Digital sensors will adjust heat output automatically to prevent damage.",
        "extra_exclusion": "- Tampering with or recalibrating the internal microprocessor or thermal sensors."
    },
    {
        "name": "BreezeLite BLD-Travel Harmony", "power": "1200W/1600W Dual Voltage", 
        "tech": "Compact Folding Design", 
        "ops": "1. Unfold handle until it clicks into place.\n2. Check the dual-voltage selector (125V/250V) on the handle to ensure it matches the local outlet.",
        "extra_exclusion": "- Damage to the folding hinge mechanism caused by forced closure."
    },
    {
        "name": "BreezeLite BLD-Titanium Glide", "power": "2000W Titanium Barrel", 
        "tech": "Direct Ion Technology", 
        "ops": "1. Select the highest heat setting for keratin-treated or thick hair.\n2. The titanium barrel provides steady heat conductivity for high-speed styling.",
        "extra_exclusion": "- Discoloration or tarnishing of the titanium barrel due to chemical hair treatments."
    },
    {
        "name": "BreezeLite BLD-Curl Master", "power": "1400W Soft-Flow Motor", 
        "tech": "Deep Bowl Diffuser Integration", 
        "ops": "1. Attach diffuser before powering on.\n2. Use Low heat and Low speed. Place curls in the diffuser and lift toward the scalp.",
        "extra_exclusion": "- Physical damage to the diffuser attachment prongs or mounting ring."
    },
    {
        "name": "BreezeLite BLD-Quiet Care", "power": "1500W Sound-Dampened Motor", 
        "tech": "Acoustic Silence Chamber", 
        "ops": "1. Use normally in any environment requiring noise sensitivity.\n2. Ensure the silence chamber vents are never covered during use.",
        "extra_exclusion": "- Degradation of sound-dampening foam caused by liquid exposure or cleaning agents."
    },
    {
        "name": "BreezeLite BLD-Digital Elite", "power": "1900W Digital Interface", 
        "tech": "LCD Temperature Display", 
        "ops": "1. Use the +/- buttons to select exact temperature.\n2. Monitor the LCD to ensure the dryer reaches the desired styling heat.",
        "extra_exclusion": "- Cracking, bleeding, or pixel failure of the LCD interface screen."
    }
]

for m in models:
    pdf = BreezeLitePDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, f"Model: {m['name']}", 0, 1, 'L')
    pdf.ln(2)

    pdf.section_title("OPERATING INSTRUCTIONS")
    pdf.section_body(
        "ALCI SAFETY TEST (BEFORE EVERY USE):\n"
        "1. Plug into a standard outlet. Press the TEST button; the RESET button should pop out.\n"
        "2. Press RESET to reactivate the unit. Do not use if the test fails.\n\n"
        "DRYING STEPS:\n" + m['ops']
    )

    pdf.section_title("MAINTENANCE & STORAGE")
    pdf.section_body(
        "1. Unplug the unit and allow it to cool completely before cleaning.\n"
        "2. Use a soft brush to remove dust and lint from the rear intake filter monthly. Blocked airflow causes overheating and potential fire hazard.\n"
        "3. DO NOT WRAP THE POWER CORD AROUND THE DRYER. Store the cord loosely coiled to prevent internal wire breakage and sparking."
    )

    pdf.section_title("LIMITED THREE (3) MONTH WARRANTY")
    pdf.section_body(
        "BreezeLite Global warrants this product against defects in workmanship and materials for a "
        "period of three (3) months from the original date of purchase. This warranty only covers "
        "defects in workmanship and materials. The warranty does not include damage due to abuse or "
        "misuse, any commercial use, or accidents."
    )

    pdf.section_title("WHAT IS NOT COVERED")
    base_exclusions = (
        "- Normal wear and tear.\n"
        "- Damage caused by misuse, abuse, or failure to follow instructions.\n"
        "- Damage caused by wrapping the power cord around the unit.\n"
        "- Use with a voltage converter or improper electrical current.\n"
        "- Damage resulting from immersion in water.\n"
    )
    pdf.section_body(base_exclusions + m['extra_exclusion'])

    pdf.section_title("LIMITATIONS")
    pdf.section_body(
        "The warranty stated above is the only warranty applicable to this product. Other expressed "
        "or implied warranties are hereby disclaimed. The manufacturer shall not be liable for incidental "
        "or consequential damages resulting from the use of this product. Any implied warranty of "
        "merchantability or fitness for a particular purpose is limited to the duration of this warranty."
    )

    pdf.section_title("HOW TO FILE A CLAIM")
    pdf.section_body(
        "For warranty service, contact BreezeLite Global Support. PHYSICAL SHIPPING IS NOT REQUIRED.\n"
        "Email: support@breezeliteglobal.com | Provide: Model #, Receipt Photo, and Video of defect."
    )

    file_name = f"{m['name'].replace(' ', '_')}.pdf"
    pdf.output(os.path.join(output_dir, file_name))

print(f"Generated 10 Hairdryer manuals in '{output_dir}'.")