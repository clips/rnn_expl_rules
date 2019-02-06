from os.path import realpath, join

class Entity:
    def __init__(self, begin = None, end = None, sem_type = None, mention = None, assertion = None, cui = None):
        self.begin = begin
        self.end = end
        self.sem_type = sem_type
        self.mention = mention
        self.assertion = assertion
        self.cui = cui


class Clamp:

    def get_entities(self, f_clamp, dir_clamp):
        entities = list() #list of Entity objects
        with open(realpath(join(dir_clamp, f_clamp))) as f:

            for line in f:
                line = line.split('\t')

                if line[0] != 'NamedEntity':
                    break  # assuming that named entities are the first lines of the file (to speed up the process)

                begin = int(line[1])
                end = int(line[2])

                sem_type = cui = assertion = mention = None

                for i in range(3, len(line), 1):
                    if line[i].startswith('semantic'):
                        sem_type = line[i].split("=")[1].strip()
                    elif line[i].startswith("assertion"):
                        assertion = line[i].split("=")[1].strip()
                    elif line[i].startswith('cui'):
                        cui = line[i].split('=')[1].strip()
                    elif line[i].startswith('ne'):
                        mention = line[i][line[i].index('=') + 1:].strip() #.replace(" ", "_")

                entities.append(Entity(begin, end, sem_type, mention, assertion, cui))


        return entities


