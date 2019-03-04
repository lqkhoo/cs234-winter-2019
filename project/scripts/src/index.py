from mujoco_py import load_model_from_path, functions
from mujoco_py.generated import const


class Index(object):
    """
    """

    def __init__(self, xml_filepath=None, mjmodel=None):
        self.mjmodel = None
        self._name2id = {}
        self._id2name = {} # Mujoco 1.5 doesn't have id2name so we construct our own index

        self._index_model(xml_filepath, mjmodel)
        self._build_index()


    def id_of(self, name, mjtobj):
        key = (name, mjtobj)
        if key not in self._name2id:
            raise KeyError()
        return self._name2id[key]


    def name_of(self, id, mjtobj):
        key = (id, mjtobj)
        if key not in self._id2name:
            raise KeyError()
        return self._id2name[key]


    def _index_model(self, mjmodel=None, xml_filepath=None):
        assert (not (mjmodel is None and xml_filepath is None))
        if mjmodel is not None:
            self.mjmodel = mjmodel
        elif xml_filepath is not None:
            self.mjmodel = load_model_from_path(xml_filepath)


    def _build_index(self):
        """
        Build index to name mappings for the parts we need.
        We make the assumption that defined names are unique. Other than
        Humanoid_CMU it works out just fine.
        """
        dic = {
            const.OBJ_BODY: (self.mjmodel.nbody, self.mjmodel.name_bodyadr),
            const.OBJ_GEOM: (self.mjmodel.ngeom, self.mjmodel.name_geomadr),
            const.OBJ_ACTUATOR: (self.mjmodel.nu, self.mjmodel.name_actuatoradr)
        }
        for mjtobj, value in dic.items():
            n = value[0]
            name_idx = value[1]
            for idx in range(n):
                name = self._name_from_idx(name_idx[idx])
                if name in self._name2id:
                    raise KeyError("Duplicate name {0}".format(name))
                self._name2id[(name, mjtobj)] = idx
                self._id2name[(idx, mjtobj)] = name


    def _name_from_idx(self, idx):
        name = b''
        char_array = self.mjmodel.names
        for i in range(idx, len(char_array)):
            name += char_array[i]
            if char_array[i] == b'':
                break
        return name.decode('utf-8')

