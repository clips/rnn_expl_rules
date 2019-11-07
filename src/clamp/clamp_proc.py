from os.path import realpath, join


class Entity:
    def __init__(self,
                 begin=None, end=None,
                 sem_type=None,
                 mention=None,
                 assertion=None, cui=None):
        self.begin = begin
        self.end = end
        self.sem_type = sem_type
        self.mention = mention
        self.assertion = assertion
        self.cui = cui

    def __repr__(self):
        representation = "\n"
        representation += ("Begin: " + str(self.begin) + "\n")
        representation += ("End: " + str(self.end) + "\n")
        representation += ("Semantic type: " + str(self.sem_type) + "\n")
        representation += ("Mention: " + str(self.mention) + "\n")
        representation += ("Assertion: " + str(self.assertion) + "\n")
        representation += ("CUI: " + str(self.cui) + "\n")
        return representation


class Relation:
    def __init__(self, ent1, ent2, rel):
        self.entity1 = ent1
        self.entity2 = ent2
        self.rel = rel

    def __repr__(self):
        representation = "\n"
        representation += ("Entity1: " + str(self.entity1) + "\n")
        representation += ("Entity2: " + str(self.entity2) + "\n")
        representation += ("Relation: " + str(self.rel) + "\n")
        return representation


class Clamp:

    def get_entities(self, f_clamp, dir_clamp):
        with open(realpath(join(dir_clamp, f_clamp))) as f:
            return self.get_entities_from_text(f.read())

    @staticmethod
    def get_entities_from_text(text):
        entities = list()  # list of Entity objects
        text = text.split("\n")
        for line in text:
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

    def get_relations_neg(self, fname, dir_clamp, dir_text):
        clamp_text = open(realpath(join(dir_clamp, fname))).read()
        text = open(realpath(join(dir_text, fname))).read()
        return self.get_relations_neg_from_text(clamp_text, text)

    @staticmethod
    def get_relations_neg_from_text(clamp_text, text):
        rels = list()  # list of Relation objects

        clamp_text = clamp_text.split("\n")

        for line in clamp_text:
            line = line.split('\t')

            if line[0] != 'Relation':
                # print("Not a relation")
                continue  # we are only interested in relations

            if line[7].split('=')[1].strip() != 'NEG_Of':
                continue  # not interested in non-negation relations
            rel_type = 'NEG_Of'

            begin1, end1, semtype1 = int(line[1]), int(line[2]), line[3]  # semtype1 is problem/treatment/test
            begin2, end2, semtype2 = int(line[4]), int(line[5]), line[6]  # semtype2 is

            ent1 = Entity(begin=begin1, end=end1, sem_type=semtype1, mention=text[begin1:end1])
            ent2 = Entity(begin=begin2, end=end2, sem_type=semtype2, mention=text[begin2:end2])

            cur_rel = Relation(ent1, ent2, rel_type)

            rels.append(cur_rel)

        return rels


if __name__ == '__main__':
    clamp_obj = Clamp()

    entities = clamp_obj.get_entities(f_clamp='1.txt',
                                      dir_clamp='/home/madhumita/dataset/sepsis_synthetic/clamp')
    rels = clamp_obj.get_relations_neg(fname='1.txt',
                                       dir_clamp='/home/madhumita/dataset/sepsis_synthetic/clamp',
                                       dir_text='/home/madhumita/dataset/sepsis_synthetic/text')