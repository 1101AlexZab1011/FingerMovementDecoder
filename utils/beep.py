import os
from typing import Union, Optional


class Beeper(object):
    def __init__(
            self,
            duration: Union[int, float, list[Union[int, float]]],
            frequency: Union[int, float, list[Union[int, float]]],
            repeat: Optional[int] = None
    ):
        self.duration = duration
        self.frequency = frequency
        self._repeat = repeat

    def __str__(self):
        return f'Beeper object called ' \
               f'{(lambda times: str(times) + " times" if times > 1 else "once")(self.repeat)} ' \
               f'at {self.frequency} Hz ' \
               f'during {self.duration} sec'

    def __call__(self):

        if not isinstance(self.frequency, list):
            frequencies = [self.frequency for _ in range(self.repeat)]
        else:
            frequencies = self.frequency
            if self.repeat is None or len(self.frequency) < self.repeat:
                self.repeat = len(self.frequency)

        if not isinstance(self.duration, list):
            durations = [self.duration for _ in range(self.repeat)]
        else:
            durations = self.duration
            if len(self.duration) < self.repeat:
                self.repeat = len(self.duration)

        for _, freq, dur in zip(range(self.repeat), frequencies, durations):
            os.system('play -nq -t alsa synth {} sine {}'.format(dur, freq))

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration: Union[int, float, list[Union[int, float]]]):
        self._duration = duration

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: Union[int, float, list[Union[int, float]]]):
        self._frequency = frequency

    @property
    def repeat(self):
        return self._repeat

    @repeat.setter
    def repeat(self, repeat: int):
        if isinstance(repeat, int):
            self._repeat = repeat
        else:
            raise AttributeError(
                f'The number of repetitions must be an integer, {type(repeat)} is given'
            )
