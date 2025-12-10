import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

#Twinkle Twinkle Little Star MIDI note values
# test_data = [60, 60, 67, 67, 69, 69, 67,65, 65, 64, 64, 62, 62, 60,
#     67, 67, 65, 65, 64, 64, 62,67, 67, 65, 65, 64, 64, 62,60, 60, 67, 67, 69, 69, 67,
#     65, 65, 64, 64, 62, 62, 60, 67, 67, 74, 74, 76, 76, 74,72, 72, 71, 71, 69, 69, 67,
#     74, 74, 72, 72, 71, 71, 69, 74, 74, 72, 72, 71, 71, 69,67, 67, 74, 74, 76, 76, 74,
#     72, 72, 71, 71, 69, 69, 67,74, 74, 72, 72, 71, 71, 69,74, 74, 72, 72, 71, 71, 69,
#     67, 67, 74, 74, 76, 76, 74, 72, 72, 71, 71, 69, 69, 67]

class MusicAnalysis:
    
    char_dict= char_notes = {
    "C_1": 0,  "CS_1": 1,  "D_1": 2,  "DS_1": 3,  "E_1": 4,  "ES_1": 5,
    "FS_1": 6, "G_1": 7,  "GS_1": 8,  "A_1": 9,  "AS_1": 10, "B_1": 11,
    "BS_1": 12, "CS0": 13, "D0": 14, "DS0": 15, "E0": 16, "ES0": 17,
    "FS0": 18, "G0": 19, "GS0": 20, "A0": 21, "AS0": 22, "B0": 23,
    "BS0": 24, "CS1": 25, "D1": 26, "DS1": 27, "E1": 28, "ES1": 29,
    "FS1": 30, "G1": 31, "GS1": 32, "A1": 33, "AS1": 34, "B1": 35,
    "BS1": 36, "CS2": 37, "D2": 38, "DS2": 39, "E2": 40, "ES2": 41,
    "FS2": 42, "G2": 43, "GS2": 44, "A2": 45, "AS2": 46, "B2": 47,
    "BS2": 48, "CS3": 49, "D3": 50, "DS3": 51, "E3": 52, "ES3": 53,
    "FS3": 54, "G3": 55, "GS3": 56, "A3": 57, "AS3": 58, "B3": 59,
    "BS3": 60, "CS4": 61, "D4": 62, "DS4": 63, "E4": 64, "ES4": 65,
    "FS4": 66, "G4": 67, "GS4": 68, "A4": 69, "AS4": 70, "B4": 71,
    "BS4": 72, "CS5": 73, "D5": 74, "DS5": 75, "E5": 76, "ES5": 77,
    "FS5": 78, "G5": 79, "GS5": 80, "A5": 81, "AS5": 82, "B5": 83,
    "BS5": 84, "CS6": 85, "D6": 86, "DS6": 87, "E6": 88, "ES6": 89,
    "FS6": 90, "G6": 91, "GS6": 92, "A6": 93, "AS6": 94, "B6": 95,
    "BS6": 96, "CS7": 97, "D7": 98, "DS7": 99, "E7": 100, "ES7": 101,
    "FS7": 102, "G7": 103, "GS7": 104, "A7": 105, "AS7": 106, "B7": 107,
    "BS7": 108, "CS8": 109, "D8": 110, "DS8": 111, "E8": 112, "ES8": 113,
    "FS8": 114, "G8": 115, "GS8": 116, "A8": 117, "AS8": 118, "B8": 119,
    "BS8": 120, "CS9": 121, "D9": 122, "DS9": 123, "E9": 124, "ES9": 125,
    "FS9": 126, "G9": 127
    }

    char_notes = (pd.DataFrame([{"note": n, "int": i} for n, i in char_dict.items()]).sort_values("int").reset_index(drop=True))

    def __init__(self, data):
        """
        Initialize the MusicAnalysis with a DataFrame containing music data and ensures
        data is in list format.
        """
        self.data = data
    
    def count_notes(self):
        """
        Count the occurrences of each note in the dataset and pair them 
        with the corresponding note names in the Note file.
        """
        note_counts = pd.Series(self.data).value_counts().sort_index()
        note_counts.index.name = 'int'
        note_counts = note_counts.reset_index(name='count')
        merged = pd.merge(self.char_notes, note_counts, on='int', how='left').fillna(0)
        merged_counts = merged[['note', 'count']].query('count > 0')
        print(merged_counts)
        return merged_counts

    def riffs(self):
        """
        Identify and count repeated sequences of notes (riffs) in the dataset.
        """
        patterns = [tuple(self.data[i:i+3]) for i in range(len(self.data)-2)]
        pattern_counts = Counter(patterns)
        max_patern = max(pattern_counts, key=pattern_counts.get)
        named_pattern = [self.char_notes.loc[self.char_notes['int'] == note, 'note'].values[0] for note in max_patern]
        print(f"Most common riff: {'-'.join(named_pattern)} with count {pattern_counts[max_patern]}")
        return pattern_counts
    
    def pitch(self):
        """
        Calculate the average note value in the dataset, and print the 2 note
        characters on either side of the average value.
        """
        avg = round(np.mean(self.data), 3)
        lo = int(np.floor(avg))
        hi = int(np.ceil(avg))
        lo_note = self.char_notes.loc[self.char_notes['int'] == lo, 'note'].iat[0]
        hi_note = self.char_notes.loc[self.char_notes['int'] == hi, 'note'].iat[0]
        print(f"Average note value is {avg} which is between {lo_note} and {hi_note}")

    def plot_music(self):
        """
        Plot a bar chart of reversed note values (127 - note) 
        in the order they appear in the sequence.
        """
        data = self.data
        x = range(len(data))
        plt.figure(figsize=(12, 4))
        plt.bar(x, data)
        plt.xticks([0, len(data)-1], ["Beginning", "End"])
        plt.yticks([0, 127], ["Low Pitch", "High Pitch"])
        plt.xlabel("Note Position")
        plt.ylabel("Pitch")
        plt.title("Pitch Plot of Song")
        plt.tight_layout()
        plt.show()

    def counts_plot(self):
        """
        Plot a bar chart of note counts from the merged DataFrame.
        """
        merged_counts = self.count_notes()
        plt.figure(figsize=(12, 6))
        plt.bar(merged_counts['note'], merged_counts['count'])
        plt.xticks(rotation=90)
        plt.xlabel("Note")
        plt.ylabel("Count")
        plt.title("Note Counts")
        plt.tight_layout()
        plt.show()

# music = MusicAnalysis(test_data)
# music.count_notes()
# music.riffs()
# music.pitch()
# music.plot_music()
# music.counts_plot()