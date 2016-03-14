import os
import cPickle


class ResultsManager(object):
    def __init__(self):
        self._path = 'results'
        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def exists(self, key):
        return os.path.exists(os.path.join(self._path, '{}.pkl'.format(key)))

    def get(self, key):
        print('Getting: {}'.format(key))
        return cPickle.load(open(os.path.join(self._path, '{}.pkl'.format(key)), 'r'))

    def save(self, key, result):
        print('Saving: {}'.format(key))
        if not self.exists(key):
            return cPickle.dump(result, open(os.path.join(self._path, '{}.pkl'.format(key)), 'w'))
        else:
            print('Result {} already exists, delete first'.format(key))



