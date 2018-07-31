class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            output_dir = '/path/to/VAR/ucf101'
            return '/path/to/UCF-101', output_dir  # folder that contains class labels
        elif database == 'hmdb51':
            output_dir = '/path/to/VAR/hmdb51'
            return '/path/to/hmdb-51', output_dir # folder that contains class labels
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError