import unittest
import os
import sys
from unittest.mock import MagicMock, patch


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)

from Music_Generation.play_song import PlaySong


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

        self.midifile_patcher = patch("Music_Generation.play_song.MidiFile")
        self.midi_track_patcher = patch("Music_Generation.play_song.MidiTrack")
        self.message_patcher = patch("Music_Generation.play_song.Message")

        self.mock_MidiFile = self.midifile_patcher.start()
        self.mock_MidiTrack = self.midi_track_patcher.start()
        self.mock_Message = self.message_patcher.start()

        # Patch pygame.mixer and pygame.time.Clock
        self.mixer_patcher = patch("Music_Generation.play_song.pygame.mixer")
        self.clock_patcher = patch("Music_Generation.play_song.pygame.time.Clock")

        self.mock_mixer = self.mixer_patcher.start()
        self.mock_clock = self.clock_patcher.start()
    
    def tearDown(self):
        """Run after each test: stop all patches."""
        self.midifile_patcher.stop()
        self.midi_track_patcher.stop()
        self.message_patcher.stop()
        self.mixer_patcher.stop()
        self.clock_patcher.stop()
        
    def test_init_calls_generate_and_play(self):
        # Patch the instance methods to avoid running real implementations
        with patch.object(PlaySong, "generate_midi") as mock_gen, \
             patch.object(PlaySong, "play_midi") as mock_play:

            song = PlaySong(self.notes, self.filename)

            mock_gen.assert_called_once_with(self.notes, self.filename)
            mock_play.assert_called_once_with(self.filename)
            
            self.assertEqual(song.notes, self.notes)
            self.assertEqual(song.name_of_file, self.filename)
            
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
if __name__ == "__main__":
    unittest.main()