import os
from mido import MidiFile

class MidiDatasetLoader:
    def __init__(self,folder_path):
        self.folder_path =folder_path
        
        self.songs = self._load_all_songs()
        
    def _load_all_songs(self):
        songs = []
        midi_files = self._get_midi_files()
        
        for path in midi_files:
            notes=self._extract_notes(path)
            if notes:
                songs.append(notes)
        return songs
    
    def _get_midi_files(self):
        files=[]
        for name in os.listdir(self.folder_path):
            if name.lower().endswith((".mid", ".midi")):
                files.append(os.path.join(self.folder_path, name))
        return files
    
    def _extract_notes(self,midi_path):
        midi = MidiFile(midi_path)
        notes=[]
        for track in midi.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity >0:
                    pitch = msg.note
                    notes.append(pitch)
        return notes