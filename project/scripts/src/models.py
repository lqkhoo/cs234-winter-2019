from mujoco_py import load_model_from_path, functions
from mujoco_py.generated import const


class Model(object):
    """
    This is a wrapper to mjModel that *stays out of the way*.
    http://mujoco.org/book/APIreference.html#mjModel
    It's meant to only encapsulate convenience functions like geomid
    to name mappings etc.

    We don't have (and don't need) something as comprehensive as
    dm_control's mujoco/indexer.py
    """

    def __init__(self, xml_filepath=None, mjmodel=None):
        self.mjmodel = None
        self.name2id = {}
        self.idntype2name = {} # Mujoco 1.5 doesn't have id2name so we construct our own index
        if mjmodel is not None:
            self.from_mjmodel(mjmodel)
        elif xml_filepath is not None:
            self.from_xml(xml_filepath)
        
    def from_mjmodel(self, mjmjmodel):
        self.mjmodel = mjmjmodel
        self._initialize()

    def from_xml(self, xml_filepath):
        self.mjmodel = load_model_from_path(xml_filepath)
        self._initialize()

    def _initialize(self):
        """Complete the rest of the initialization steps"""
        self._build_index()

    def _build_index(self):
        """Build index to name mappings for important parts"""
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
                if name in self.name2id:
                    raise KeyError("Duplicate name {0}".format(name))
                self.name2id[name] = (idx, mjtobj)
                self.idntype2name[(idx, mjtobj)] = name

    def _name_from_idx(self, idx):
        name = b''
        char_array = self.mjmodel.names
        for i in range(idx, len(char_array)):
            name += char_array[i]
            if char_array[i] == b'':
                break
        return name.decode('utf-8')


class PointRoller(Model):
    MODEL_PATH = '../assets/pointroller.xml'
    def __init__(self, xml_filepath=MODEL_PATH, mjmodel=None):
        super().__init__(xml_filepath, mjmodel)

