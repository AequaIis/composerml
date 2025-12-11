import unittest
import os
import sys
from unittest.mock import MagicMock, patch


from composerml.music_generation import PlaySong


class TestPlaySong(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes = [60, 62, 64]  # simple C major fragment
        cls.filename = "test_song.mid"
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.filename):
            os.remove(cls.filename)
    
    def setUp(self):
        """Run before each test: patch heavy side-effect dependencies. Patches are there so we are not using the real dependancies"""

        self.midifile_patcher = patch("composerml.music_generation.play_song.MidiFile")
        self.midi_track_patcher = patch("composerml.music_generation.play_song.MidiTrack")
        self.message_patcher = patch("composerml.music_generation.play_song.Message")

        self.mock_MidiFile = self.midifile_patcher.start()
        self.mock_MidiTrack = self.midi_track_patcher.start()
        self.mock_Message = self.message_patcher.start()

        # Patch pygame.mixer and pygame.time.Clock
        self.mixer_patcher = patch("composerml.music_generation.play_song.pygame.mixer")
        self.clock_patcher = patch("composerml.music_generation.play_song.pygame.time.Clock")

        self.mock_mixer = self.mixer_patcher.start()
        self.mock_clock = self.clock_patcher.start()
    
    def tearDown(self):
        """Run after each test: stop all patches."""
        self.midifile_patcher.stop()
        self.midi_track_patcher.stop()
        self.message_patcher.stop()
        self.mixer_patcher.stop()
        self.clock_patcher.stop()
        
    def test_init(self):
        song = PlaySong()
        self.assertIsInstance(song, PlaySong)
            
    def test_generate_midi_creates_track_and_messages(self):
        
        song = PlaySong.__new__(PlaySong)

        song.generate_midi(self.notes, self.filename)

        self.mock_MidiFile.assert_called_once_with()
        self.mock_MidiTrack.assert_called_once_with()
        new_mid_instance = self.mock_MidiFile.return_value
        new_track_instance = self.mock_MidiTrack.return_value

        new_mid_instance.tracks.append.assert_called_once_with(new_track_instance)

        # Ensure Message(note_on, note, velocity, time) was created for each note
        calls = self.mock_Message.call_args_list
        self.assertEqual(len(calls), len(self.notes))
        for call, note in zip(calls, self.notes):
            args, kwargs = call
            self.assertEqual(args[0], "note_on")
            self.assertEqual(kwargs["note"], note)
            self.assertEqual(kwargs["velocity"], 64)
            self.assertEqual(kwargs["time"], 128)

        # Ensure MidiFile.save was called with the filename
        new_mid_instance.save.assert_called_once_with(self.filename)

    def test_play_midi_uses_pygame_mixer_correctly(self):
        self.mock_mixer.music.get_busy.side_effect = [True, False]
        clock_instance = self.mock_clock.return_value

        song = PlaySong.__new__(PlaySong)
        song.play_midi(self.filename)

        self.mock_mixer.init.assert_called_once()
        self.mock_mixer.music.load.assert_called_once_with(self.filename)
        self.mock_mixer.music.play.assert_called_once()

        self.assertGreaterEqual(self.mock_mixer.music.get_busy.call_count, 1)

        clock_instance.tick.assert_called_with(10)
    
    def generate_play_midi_test(self):
        """Integration test: generate a midi file and play it (with patched pygame)"""
        song = PlaySong()
        song.generate_midi(self.notes, self.filename)
        song.play_midi(self.filename)

if __name__ == "__main__":
    unittest.main()