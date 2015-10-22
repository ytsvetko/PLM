#!/usr/bin/env python3

import glob
import os

FEATURES=[
("Consonant Inventories", "Moderately small"),
("Consonant Inventories", "Small"),
("Consonant Inventories", "Average"),
("Consonant Inventories", "Moderately large"),
("Consonant Inventories", "Large"),
("Vowel Quality Inventories", "Small"),
("Vowel Quality Inventories", "Average"),
("Vowel Quality Inventories", "Large"),
("Consonant-Vowel Ratio", "Moderately low"),
("Consonant-Vowel Ratio", "Low"),
("Consonant-Vowel Ratio", "Average"),
("Consonant-Vowel Ratio", "Moderately high"),
("Consonant-Vowel Ratio", "High"),
("Voicing in Plosives and Fricatives", "No voicing contrast"),
("Voicing in Plosives and Fricatives", "Voicing contrast in plosives alone"),
("Voicing in Plosives and Fricatives", "Voicing contrast in fricatives alone"),
("Voicing in Plosives and Fricatives", "Voicing contrast in both plosives and fricatives"),
("Voicing and Gaps in Plosive Systems", "Other"), 
("Voicing and Gaps in Plosive Systems", "/p t k b d g/"),
("Voicing and Gaps in Plosive Systems", "Missing /p/"),
("Voicing and Gaps in Plosive Systems", "Missing /g/"),
("Voicing and Gaps in Plosive Systems", "Both missing"),
("Uvular Consonants", "No uvulars"),
("Uvular Consonants", "Uvular stops only"),
("Uvular Consonants", "Uvular continuants only"),
("Uvular Consonants", "Uvular stops and continuants"),
("Glottalized Consonants", "No glottalized consonants"),
("Glottalized Consonants", "Ejectives only"),
("Glottalized Consonants", "Implosives only"),
("Glottalized Consonants", "Glottalized resonants only"),
("Glottalized Consonants", "Ejectives, implosives and glottalized resonants"),
("Glottalized Consonants", "Ejectives and implosives"),
("Glottalized Consonants", "Ejectives and glottalized resonants"),
("Glottalized Consonants", "Implosives and glottalized resonants"),
("Lateral Consonants", "No laterals"),
("Lateral Consonants", "Laterals, but no /l/, no obstruent lateral"),
("Lateral Consonants", "/l/, no obstruent laterals"),
("Lateral Consonants", "/l/ and lateral obstruents"),
("Lateral Consonants", "No /l/, but lateral obstruents"),
("The Velar Nasal", "Velar nasal, also initially"),
("The Velar Nasal", "Velar nasal, but not initially"),
("The Velar Nasal", "No velar nasal"),
("Vowel Nasalization", "Contrastive nasal vowels present"),
("Vowel Nasalization", "Contrastive nasal vowels absent"),
("Nasal Vowels in West Africa", "no nasal vs. oral vowel contrast"),
("Nasal Vowels in West Africa", "two-way nasal vs. oral vowel contrast (/ṽ/ vs. /V/) without nasal spreading"),
("Nasal Vowels in West Africa", "two-way nasal vs. oral vowel contrast (/ṽ/ vs. /V/) with nasal spreading"),
("Nasal Vowels in West Africa", "four-way nasal vs. oral vowel contrast (/ṽ/ vs. /ṽː/ vs. /V/ vs. /Vː/) without nasal spreading"),
("Nasal Vowels in West Africa", "four-way nasal vs. oral vowel contrast (/ṽ/ vs. /ṽː/ vs. /V/ v /Vː/) with nasal spreading"),
("Front Rounded Vowels", "None"),
("Front Rounded Vowels", "High and mid"),
("Front Rounded Vowels", "High only"),
("Front Rounded Vowels", "Mid only"),
("Syllable Structure", "Simple syllable structure"),
("Syllable Structure", "Moderately complex syllable structure"),
("Syllable Structure", "Complex syllable structure"),
("Tone", "No tones"),
("Tone", "Simple tone system"),
("Tone", "Complex tone system"),
("Fixed Stress Locations", "No fixed stress)"),
("Fixed Stress Locations", "Initial"),
("Fixed Stress Locations", "Second"),
("Fixed Stress Locations", "Antepenultimate"),
("Fixed Stress Locations", "Penultimate"),
("Fixed Stress Locations", "Ultimate"),
("Weight-Sensitive Stress", "Left-edge"),
("Weight-Sensitive Stress", "Left-oriented"),
("Weight-Sensitive Stress", "Right-edge"),
("Weight-Sensitive Stress", "Right-oriented"),
("Weight-Sensitive Stress", "Unbounded"),
("Weight-Sensitive Stress", "Combined"),
("Weight-Sensitive Stress", "Not predictable"),
("Weight-Sensitive Stress", "Fixed stress"),
("Weight Factors in Weight", "No weight, or weight factor unknown"),
("Weight Factors in Weight", "Long vowel"),
("Weight Factors in Weight", "Coda consonant"),
("Weight Factors in Weight", "Long vowel + Coda"),
("Weight Factors in Weight", "Prominence"),
("Weight Factors in Weight", "Lexical"),
("Weight Factors in Weight", "Combined"),
("Rhythm Types", "Trochaic"),
("Rhythm Types", "Iambic"),
("Rhythm Types", "Undetermined"),
("Rhythm Types", "Absent"),
("Absence of Common Consonants", "No bilabials or nasals"),
("Absence of Common Consonants", "No fricatives or nasals"),
("Absence of Common Consonants", "All present"),
("Absence of Common Consonants", "No bilabials"),
("Absence of Common Consonants", "No fricatives"),
("Absence of Common Consonants", "No nasals"),
("Presence of Uncommon Consonants", "None"),
("Presence of Uncommon Consonants", "Clicks, pharyngeals, and 'th'"),
("Presence of Uncommon Consonants", "Pharyngeals and 'th'"),
("Presence of Uncommon Consonants", "Clicks"),
("Presence of Uncommon Consonants", "Labial-velars"),
("Presence of Uncommon Consonants", "Pharyngeals"),
("Presence of Uncommon Consonants", "'Th' sounds"),
]

FEATURES = [(feat.lower(), val.lower()) for (feat, val) in FEATURES]

def BuildFeatureVector(in_filename, out_filename):
  in_f = open(in_filename).readlines()
  out_f = open(out_filename, "w")
  found = False
  for (feat, val) in FEATURES:
    for line in in_f:
      line = line.lower()
      if feat in line and val in line and not found:
        out_f.write("1\n")
        found = True
        break
    if not found:
      out_f.write("0\n")
    found = False
    
for in_filename in glob.glob("/usr1/home/ytsvetko/projects/mnlm/data/wals/csv/*.csv"):
  print(in_filename)
  lang = os.path.basename(in_filename)[:-4]
  out_filename = "/usr1/home/ytsvetko/projects/mnlm/data/wals/csv/"+lang+".txt"
  BuildFeatureVector(in_filename, out_filename)

