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


def custom_generator(list_of_values):
    for i in list_of_values:
        yield i+1


def test_IteratorReStarter():
    new = IteratorReStarter(CustomIterator, [1])
    # run over the iterator once. An iterator will remain empty after one full iteration.
    list(new)
    assert list(new) != []

    new = IteratorReStarter(custom_generator, [[1, 2, 3, 4]])
    # run over the generator once. An generator will remain empty after one full iteration.
    list(new)
    assert list(new) != []
