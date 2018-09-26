from ariels_utils.ariels_utils import IteratorReStarter

class CustomIterator(object):
    """Custom Iterator for testing"""
    def __init__(self, x):
        self.x = x
        self.max_x = 100

    def __iter__(self):
        return self

    def __next__(self):
        if self.x < self.max_x:
            self.x += 1
            return self.x
        else:
            raise StopIteration


def test_IteratorReStarter():
    iterator_object = CustomIterator(12)
    new = IteratorReStarter(CustomIterator, [1])
    # run over the iterator once. An iterator will remain empty after one full iteration.
    list(new)
    assert list(new) != []
