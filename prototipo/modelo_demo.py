# -*- coding: utf-8 -*-
"""
MODELO DE DEMOSTRACIÓN: se entrena con epicrisis INVENTADAS, generadas aquí por plantillas con semilla fija. No contiene
nada de MIMIC-IV y no tiene valor clínico: sirve para que el sistema corra sin credencial (clase, docente, Docker).

Usa la misma función de entrenamiento y la misma configuración que el modelo real (prototipo.entrenar_local), de modo
que el flujo completo (contrato, huella, ficha, detección) se prueba igual. Las etiquetas también son inventadas: una
epicrisis generada con una frase de evento lleva y = 1.

También genera los ERSP inventados que la pestaña de priorización usa para mostrar la Matriz con un semestre de datos.

Uso:  python -m prototipo.modelo_demo
"""
import numpy as np
import pandas as pd

from .entrenar_local import entrenar
from .rutas import ruta_modelo_demo

SEMILLA = 42
N_EPICRISIS = 600

# (servicio, quirúrgico, motivo de consulta, enfermedad actual, curso base, diagnóstico)
ADMISIONES = [
    ("MED", 0, "Cough and fever", "presented with three days of productive cough, fever and shortness of breath; chest x-ray showed a right lower lobe infiltrate.",
     "Treated with ceftriaxone and azithromycin with steady improvement in oxygenation.", "Community-acquired pneumonia"),
    ("CMED", 0, "Shortness of breath", "presented with worsening dyspnea, orthopnea and leg swelling over one week.",
     "Diuresed with intravenous furosemide with good urine output and improvement of symptoms.", "Acute on chronic systolic heart failure"),
    ("MED", 0, "Wheezing", "presented with increased wheezing, cough and sputum production.",
     "Treated with nebulized bronchodilators, prednisone and a short course of antibiotics.", "COPD exacerbation"),
    ("MED", 0, "Leg redness", "presented with redness, warmth and swelling of the right lower leg.",
     "Treated with intravenous cefazolin; the erythema receded over several days.", "Right lower extremity cellulitis"),
    ("ORTHO", 1, "Right hip pain", "with long-standing right hip osteoarthritis was admitted for elective total hip arthroplasty.",
     "Underwent right total hip arthroplasty; physical therapy started on postoperative day one.", "Right hip osteoarthritis status post total hip arthroplasty"),
    ("SURG", 1, "Right upper quadrant pain", "presented with recurrent right upper quadrant pain after meals; ultrasound showed gallstones.",
     "Underwent laparoscopic cholecystectomy; the diet was advanced as tolerated.", "Symptomatic cholelithiasis"),
    ("MED", 0, "Nausea and vomiting", "with type 1 diabetes presented with nausea, vomiting and an elevated anion gap.",
     "Treated with an insulin infusion and intravenous fluids; the anion gap closed within a day.", "Diabetic ketoacidosis"),
    ("MED", 0, "Black stools", "presented with melena and lightheadedness; hemoglobin was low on arrival.",
     "Upper endoscopy showed a duodenal ulcer that was treated endoscopically; started on a proton pump inhibitor.", "Upper gastrointestinal bleed from duodenal ulcer"),
    ("NMED", 0, "Left arm weakness", "presented with sudden left arm weakness and slurred speech.",
     "MRI confirmed a small right middle cerebral artery infarct; started on aspirin and a statin.", "Acute ischemic stroke"),
    ("ORTHO", 1, "Left knee pain", "with severe left knee osteoarthritis was admitted for elective total knee arthroplasty.",
     "Underwent left total knee arthroplasty and worked daily with physical therapy.", "Left knee osteoarthritis status post total knee arthroplasty"),
    ("CSURG", 1, "Exertional chest pain", "with three-vessel coronary disease on catheterization was admitted for bypass surgery.",
     "Underwent coronary artery bypass grafting and was extubated on the day of surgery.", "Coronary artery disease status post bypass grafting"),
    ("CMED", 0, "Chest pain", "presented with atypical chest pain; serial troponins were negative.",
     "A stress test showed no inducible ischemia.", "Atypical chest pain, myocardial infarction ruled out"),
]

EVENTOS = [
    ["On hospital day {d} the patient developed a catheter-associated bloodstream infection; blood cultures grew {org} and the central line was removed. Completed a course of intravenous {abx}.",
     "The course was complicated by a central line-associated bloodstream infection with {org}; the line was removed and {abx} was given."],
    ["The patient had an unwitnessed fall from bed on hospital day {d} and was found on the floor; imaging showed a {fx} fracture that was managed by orthopedics.",
     "On hospital day {d} the patient fell while walking to the bathroom without assistance and sustained a {fx} fracture."],
    ["The course was complicated by vancomycin-induced acute kidney injury; vancomycin was held and creatinine returned toward baseline.",
     "Creatinine rose after starting vancomycin, consistent with drug-induced nephrotoxicity; the drug was discontinued due to the kidney injury."],
    ["The postoperative course was complicated by a surgical site infection requiring bedside incision and drainage and antibiotics.",
     "On postoperative day {d} the wound was erythematous with purulent drainage, consistent with surgical site infection; the incision was opened and packed."],
    ["A stage {st} pressure ulcer developed on the sacrum during the admission; wound care nursing was consulted.",
     "Skin examination on hospital day {d} revealed a new stage {st} pressure injury over the heel, which was offloaded and dressed."],
    ["The patient developed an anaphylactic reaction to {drug} with hypotension and wheezing, treated with intramuscular epinephrine and steroids.",
     "Shortly after the first dose of {drug} the patient had an allergic reaction with diffuse hives and hypotension requiring epinephrine."],
    ["Hospital-acquired pneumonia developed on hospital day {d} with new fever and a new infiltrate, treated with broad-spectrum antibiotics.",
     "The course was complicated by ventilator-associated pneumonia requiring a prolonged course of antibiotics."],
    ["After central venous catheter placement the patient developed an iatrogenic pneumothorax requiring chest tube placement.",
     "A post-procedural pneumothorax was noted after thoracentesis and a chest tube was placed."],
    ["Heparin-induced thrombocytopenia was diagnosed with a positive antibody test; heparin was stopped and argatroban was started.",
     "Platelets fell significantly on heparin and the HIT antibody was positive; anticoagulation was switched to argatroban."],
    ["Post-procedural hemorrhage after {proc} required transfusion of {n} units of packed red blood cells.",
     "The patient developed a large hematoma after {proc} and required transfusion and a return to the operating room."],
    ["A catheter-associated urinary tract infection developed on hospital day {d}; the Foley catheter was removed and antibiotics were given.",
     "The course was complicated by CAUTI with fever and pyuria; the urinary catheter was removed."],
    ["Due to a medication administration error the patient received a double dose of {drug2}, resulting in {efecto} treated with {antidoto}.",
     "The patient was inadvertently given {drug2} twice, leading to {efecto} that required {antidoto}."],
]

# Frases neutras: las primeras caben en cualquier epicrisis; las demás contradirían una frase de evento
NEUTRAS = ["Pain was controlled with oral medications.", "Home medications were resumed at discharge.",
           "Physical therapy cleared the patient for discharge.", "Electrolytes were repleted as needed.",
           "Vital signs were reviewed every shift.", "The diet was advanced as tolerated.",
           "Case management arranged follow-up with the primary care physician.",
           "Deep vein thrombosis prophylaxis was given with subcutaneous heparin.",
           "Glucose was managed with sliding scale insulin.", "The patient was counseled on smoking cessation.",
           "Oxygen was weaned to room air.", "A dietitian reviewed the meal plan with the patient.",
           "Follow-up was scheduled in clinic in two weeks.", "The patient and family agreed with the discharge plan.",
           "Nursing reinforced education on the new medications.", "Fall precautions were maintained during the stay."]
NEUTRAS_SIN_EVENTO = ["There were no complications during the admission.", "Blood cultures remained negative.",
                      "The patient was ambulating independently at discharge.",
                      "The patient remained hemodynamically stable throughout."]
NEUTRAS_QUIRURGICAS = ["The postoperative course was uneventful.", "The patient tolerated the procedure well."]
SOLO_QUIRURGICOS = {3}          # índice en EVENTOS de la infección de sitio quirúrgico

RELLENO = {"org": ["Staphylococcus aureus", "Klebsiella pneumoniae", "Candida albicans", "Enterococcus faecalis"],
           "abx": ["vancomycin", "cefepime", "micafungin", "daptomycin"], "fx": ["left hip", "right wrist", "nasal bone", "rib"],
           "st": ["2", "3"], "drug": ["cefazolin", "piperacillin-tazobactam", "iodinated contrast", "ciprofloxacin"],
           "proc": ["the femoral catheterization", "the renal biopsy", "the paracentesis", "the liver biopsy"], "n": ["2", "3", "4"],
           "drug2": ["insulin glargine", "metoprolol", "oxycodone", "warfarin"],
           "efecto": ["severe hypoglycemia", "bradycardia and hypotension", "respiratory depression", "a supratherapeutic INR"],
           "antidoto": ["intravenous dextrose", "glucagon", "naloxone", "vitamin K"]}

DESTINOS = ["HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "REHAB"]
PROCEDENCIAS = ["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER FROM HOSPITAL", "CLINIC REFERRAL"]
SEGUROS = ["Medicare", "Medicaid", "Private", "Other"]


def _rellenar(rng, frase):
    claves = {k: rng.choice(v) for k, v in RELLENO.items()}
    return frase.format(d=int(rng.integers(2, 9)), **claves)


NOMBRE_SERVICIO = {"MED": "MEDICINE", "CMED": "CARDIOLOGY", "NMED": "NEUROLOGY", "ORTHO": "ORTHOPAEDICS",
                   "SURG": "SURGERY", "CSURG": "CARDIAC SURGERY"}
CONDICIONES = ["Stable.", "Good.", "Stable, ambulating with assistance.", "Good, ambulating independently."]
DISPOSICIONES = {"HOME": "Home.", "HOME HEALTH CARE": "Home with services.",
                 "SKILLED NURSING FACILITY": "Extended care facility.", "REHAB": "Rehabilitation facility."}


def generar_epicrisis(n=N_EPICRISIS, semilla=SEMILLA):
    """Tabla de epicrisis inventadas con las 18 variables, el texto y la etiqueta inventada y.

    El largo y la estructura de la nota no dependen de la etiqueta: las dos clases llevan las mismas secciones y un
    número de frases parecido, para que el modelo aprenda de las frases de evento y no del largo. Las variables de la
    hospitalización tienen una asociación débil con la etiqueta."""
    rng = np.random.default_rng(semilla)
    filas = []
    for _ in range(n):
        srv, quir, cc, hpi, curso, dx = ADMISIONES[int(rng.integers(len(ADMISIONES)))]
        y = int(rng.random() < 0.5)
        edad = int(rng.integers(25, 91))
        sexo = str(rng.choice(["Mujer", "Hombre"]))
        n_frases = int(rng.integers(6, 11))
        resto = []
        if y:
            posibles = [k for k in range(len(EVENTOS)) if quir or k not in SOLO_QUIRURGICOS]
            for e in rng.choice(posibles, size=1 + int(rng.random() < 0.2), replace=False):
                resto.append(_rellenar(rng, str(rng.choice(EVENTOS[int(e)]))))
            neutras = NEUTRAS
        else:
            neutras = NEUTRAS + NEUTRAS_SIN_EVENTO + (NEUTRAS_QUIRURGICAS if quir else [])
        resto += [str(x) for x in rng.choice(neutras, size=max(2, n_frases - 2 * len(resto)), replace=False)]
        rng.shuffle(resto)
        destino = str(rng.choice(DESTINOS, p=[0.4, 0.25, 0.22, 0.13] if not y else [0.32, 0.25, 0.28, 0.15]))
        procedimiento = f"Major Surgical or Invasive Procedure: {dx.split('status post ')[-1].capitalize()}.\n\n" if quir else ""
        texto = (f"Invented discharge summary for demonstration. No real patient.\n\nService: {NOMBRE_SERVICIO[srv]}\n\n"
                 f"Chief Complaint: {cc}.\n\n{procedimiento}"
                 f"History of Present Illness: A {edad}-year-old {'woman' if sexo == 'Mujer' else 'man'} {hpi}\n\n"
                 f"Brief Hospital Course: {' '.join([curso] + resto)}\n\nDischarge Diagnosis: {dx}.\n\n"
                 f"Discharge Disposition: {DISPOSICIONES[destino]}\n\nDischarge Condition: {rng.choice(CONDICIONES)}\n")
        urgencia = not quir or rng.random() < 0.3
        procedencia = "EMERGENCY ROOM" if urgencia and rng.random() < 0.8 else str(rng.choice(PROCEDENCIAS[1:]))
        uci = int(rng.random() < (0.3 if y else 0.2))
        filas.append({
            "text": texto, "y": y, "edad": edad, "sexo": sexo,
            "horas_emergencia": float(np.round(rng.uniform(2, 14), 1)) if procedencia == "EMERGENCY ROOM" else np.nan,
            "tipo_ingreso": "ELECTIVE" if not urgencia else str(rng.choice(["EW EMER.", "URGENT", "OBSERVATION ADMIT"])),
            "procedencia": procedencia, "seguro": str(rng.choice(SEGUROS)), "servicio": srv,
            "servicio_alta": srv if rng.random() < 0.85 else "MED", "n_servicios": 1 + int(rng.random() < (0.25 if y else 0.15)),
            "servicio_quirurgico": quir, "estancia_dias": float(np.round(rng.uniform(1.5, 9) + (rng.uniform(0, 3) if y else 0), 1)),
            "n_traslados": int(rng.integers(1, 3)) + uci, "n_unidades": 1 + uci + int(rng.random() < 0.3), "uci": uci,
            "dias_uci": float(np.round(rng.uniform(0.5, 4), 1)) if uci else 0.0, "destino_alta": destino,
            "n_medicamentos": int(rng.integers(8, 30)) + (int(rng.integers(0, 5)) if y else 0),
            "n_cultivos": int(rng.integers(0, 4)) + (int(rng.integers(0, 2)) if y else 0),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- ERSP inventados para la Matriz
# (Id. del Anexo N.° 02, proceso del num. 19.6, servicio, peso relativo de aparición)
ERSP_DEMO = [
    (202, "Hospitalización", "Medicina Interna", 6), (2017, "Hospitalización", "Medicina Interna", 5),
    (2018, "Hospitalización", "Medicina Interna", 2), (601, "Hospitalización", "Unidad de Cuidados Intensivos", 3),
    (607, "Hospitalización", "Unidad de Cuidados Intensivos", 2), (206, "Hospitalización", "Medicina Interna", 4),
    (7021, "Hospitalización", "Medicina Interna", 3), (705, "Hospitalización", "Cardiología", 2),
    (604, "Quirúrgica", "Cirugía General", 4), (8018, "Quirúrgica", "Cirugía General", 2),
    (8016, "Quirúrgica", "Cirugía de Tórax", 1), (202, "Quirúrgica", "Traumatología", 2),
    (202, "Emergencia", "Emergencia", 3), (7021, "Emergencia", "Emergencia", 2), (402, "Emergencia", "Emergencia", 2),
    (705, "Ambulatorio", "Consulta Externa", 2),
]


def ersp_demo(fecha_referencia, n=48, semilla=SEMILLA):
    """ERSP confirmados inventados, con fechas dentro de los 178 días que terminan en la fecha de referencia. Cada uno de
    los 16 grupos tiene al menos un ERSP: los de índice par, uno en los 14 días que terminan en esa fecha, para que la
    Matriz tenga filas; los de índice impar, uno fuera del periodo de corte. El resto se sortea por peso, fuera del
    periodo de corte."""
    rng = np.random.default_rng(semilla)
    fin = pd.Timestamp(fecha_referencia).normalize()
    pesos = np.array([g[3] for g in ERSP_DEMO], dtype=float)
    filas = []
    for k, (id_ev, proceso, servicio, _) in enumerate(ERSP_DEMO):     # un ERSP por grupo
        dias = int(rng.integers(0, 14)) if k % 2 == 0 else int(rng.integers(15, 178))
        filas.append((id_ev, proceso, servicio, fin - pd.Timedelta(days=dias)))
    for _ in range(n - len(filas)):
        id_ev, proceso, servicio, _ = ERSP_DEMO[int(rng.choice(len(ERSP_DEMO), p=pesos / pesos.sum()))]
        filas.append((id_ev, proceso, servicio, fin - pd.Timedelta(days=int(rng.integers(15, 178)))))
    return pd.DataFrame(filas, columns=["id_ev", "proceso", "servicio", "fecha"])


def main():
    d = generar_epicrisis()
    ruta = ruta_modelo_demo()
    print(f"Entrenando el modelo de demostración con {len(d)} epicrisis inventadas ({int(d.y.sum())} con evento) ...", flush=True)
    ficha = entrenar(d.drop(columns="y"), d.y.values, ruta, "demostracion",
                     {"datos": "epicrisis inventadas por plantillas (prototipo/modelo_demo.py)", "semilla": SEMILLA})
    print(f"Listo: {ruta}\n  SHA-256: {ficha['sha256']}")


if __name__ == "__main__":
    main()
