import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.filters import Filter
import weka.core.serialization as serialization

import math
from itertools import combinations
from os.path import join, realpath

def start_jvm():
    jvm.start()

def stop_jvm():
    jvm.stop()

def load_data(fname, dir_in = "../data/", incremental = False):
    """
    Loads data in weka format
    :param fname: filename for data
    :param dir_in: input directory
    :param incremental: True to read data incrementally.
    :return: The data and the loader object
    """
    loader = Loader(classname="weka.core.converters.ArffLoader")
    if incremental:
        data = loader.load_file(realpath(join(dir_in, fname)), incremental=incremental)
    else:
        data = loader.load_file(realpath(join(dir_in, fname)))
    data.class_is_last() #Required to specify which attribute is class attribute. For us, it is the last attribute.

    return data, loader

def merge_classes(data, idx_to_merge):
    """
    :param data: The data file to filter
    :param idx_to_merge: String representation of class indices to merge 
    :return: filtered data
    """
    merge_filter = Filter(classname="weka.filters.unsupervised.attribute.MergeManyValues",
                          options=["-C", "last", "-R", idx_to_merge, "-unset-class-temporarily"])
    merge_filter.inputformat(data)
    filtered_data = merge_filter.filter(data)
    return filtered_data


def get_classifier(classifier, min_no):
    """
    Return the classifier object given the options
    :param classifier: string name of the classifier to use
    :param min_no: Minimum number of instances correctly covered by the classifier
    :return: classifier object
    """

    if classifier.lower() == 'jrip':
        cls = Classifier(classname="weka.classifiers.rules.JRip")
        options = list()
        options.append("-N")
        options.append(str(min_no))

    elif classifier.lower() in ['dec_tree', 'part']:
        if classifier.lower() == 'dec_tree':
            cls = Classifier(classname="weka.classifiers.trees.J48")
        elif classifier.lower() == 'part':
            cls = Classifier(classname="weka.classifiers.rules.PART")
        options = list()
        options.append("-M")
        options.append(str(min_no))
    else:
        raise ValueError("Please enter the correct classifier name (jrip | dec_tree | part)")


    cls.options = options
    return cls

def build_classifier(data, cls, incremental = False, loader = None):
    """
    Build classifier from the corresponding data
    :param data: weka data object
    :param cls: classifier object
    :param incremental: True if data is loaded incrementally
    :param loader: if incremental, the loader to load data
    :return: classifier
    """

    if incremental and loader is None:
        raise ValueError("Please enter a dataloader if incremental model")

    cls.build_classifier(data)

    if incremental:
        for inst in loader:
            cls.update_classifier(inst)

    return cls

def evaluate_classifier(cls, train_data, test_data):
    """
    Evaluation
    :param cls: trained classifier
    :param train_data: data to initialize priors with
    :return: evaluation object
    """
    evl = Evaluation(train_data)
    evl.test_model(cls, test_data)

    return evl

def optimize_rule_params(classifier, train_data, val_data, test_data, incremental, train_dl, metric):
    """
    Iterate over different parameter values and train a rule induction model. The best parameters are retained.
    :param classifier: string name of the classification algorithm to use
    :param train_data: Data to use for inducing rules
    :param val_data: Data for tuning hyperparameters
    :param test_data: Data for testing rule transferrability
    :param incremental: True if data is loaded incrementally
    :param train_dl: Data loader object if incremental is True
    :param metric: metric to evaluate the rule induction model using
    :return the final model for rule induction
    """
    stats = train_data.attribute_stats(train_data.class_index)
    min_inst = min(stats.nominal_counts)

    #works for binary setup
    if stats.nominal_counts[0] == min_inst:
        class_index = 0
    elif stats.nominal_counts[1] == min_inst:
        class_index = 1

    print("Number of instances in the minority class with index {}: {}".format(class_index, min_inst))

    if min_inst <= 2:
        print("Skipping this class because there are too few instances")
        return

    print("Optimizing over parameters of explaining model")

    best_n, best_model, best_eval, best_score = None, None, None, None

    # start_n = 2
    # stop_n = 11 if min_inst >= 10 else min_inst
    # step = 1
    start_n = int(min_inst/1000)
    stop_n = int(min_inst/100)
    step = start_n

    for n in range(start_n, stop_n, step):
    # for n in range(300, 301, 1):

        print("Minimum correctly covered instances: ", n)

        cls = get_classifier(classifier, n)
        cls = build_classifier(train_data, cls, incremental, train_dl)

        evl = evaluate_classifier(cls, train_data, val_data)
        cur_score = _get_score(metric, evl, class_index)

        if math.isnan(cur_score):
            break  # don't iterate to higher N value if current value covers zero instances for any class.

        if best_eval is None or cur_score >= best_score:
            best_model = cls
            best_eval = evl
            best_n = n
            best_score = cur_score

            print("Current best model:", best_model)

    if best_eval is None:
        print("Optimized model not found")
        return

    test_evl = evaluate_classifier(best_model, train_data, test_data)
    test_score = _get_score(metric, test_evl, class_index)

    print("Final results: ")
    print("Best performance found for N {}".format(best_n))
    print("Best model: ", best_model)

    print("\n Validation set results: ", best_eval.summary())
    # print("Precision, recall, F-score for the minority class: ",
    #       best_eval.precision(class_index),
    #       best_eval.recall(class_index),
    #       best_score)
    print("Optimized metric {} on validation set {}: ".format(metric, _get_score(metric, best_eval, class_index)))
    print("Validation set confusion matrix: \n", best_eval.confusion_matrix)

    print("\n Test results:", test_evl.summary())
    print("Test score for metric {}: {}".format(metric, test_score))
    print("Test confusion matrix:", test_evl.confusion_matrix)

    return best_model

def save_model(model, fname, dir_name):
    outfile = realpath(join(dir_name, fname)) #"j48.model"
    serialization.write(outfile, model)

def load_model(fname, dir_name):
    outfile = realpath(join(dir_name, fname))
    model = Classifier(jobject=serialization.read(outfile))
    print(model)
    return model

def get_rule_covering_inst(classifier, data, inst_idx):
    '''
    Finds the rule in a learned JRIP model that covers an instance
    :param classifier: trained JRIP model
    :param data: weka dataset
    :param inst_idx: instance ID to find corresponding rule of
    '''
    merge_filter = Filter(classname="weka.filters.supervised.attribute.ClassOrder",
                          options=["-C", "0"])
    merge_filter.inputformat(data)
    ordered_data = merge_filter.filter(data)

    rset = classifier.jwrapper.getRuleset()
    for i in range(rset.size()):
        r = rset.get(i)
        if r.covers(data.get_instance(inst_idx).jobject):
            print("Instance is covered by current rule:", str(r.toString(ordered_data.class_attribute.jobject)))
            break

def _get_score(metric, evl, class_index):
    if metric == 'class-f-score':
        score = evl.f_measure(class_index)
        print("F-score for the minority class: {} \n".format(score))
    elif metric == 'macro-f-score':
        score = evl.unweighted_macro_f_measure
        print("Unweighted macro f-measure: {} \n".format(score))
    elif metric == 'wt-prec':
        score = evl.weighted_precision
        print("Weighted precision: {} \n".format(score))
    elif metric == 'class-prec':
        score = evl.precision(class_index)
        print("Precision for the minority class: {} \n".format(score))
    else:
        raise ValueError("Please enter the correct metric (class-f-score|macro-f-score|wt-prec|class-prec)")

    return score

def induce_explanations(classifier, opt_metric, train_data, val_data, test_data, data_dir, out_fname, dir_out):
    """
    Induce the rules using RIPPERk (JRIP) or trees using C4.5
    :param classifier: string name of classification method to use
    :param opt_metric: metric to optimize (prec/f-score)
    :param train_data: File containing training data in arff format
    :param val_data: File containing validation data in arff format
    :param test_data: File containing test data in arff format
    :param data_dir: directory path for input file
    """

    start_jvm()

    try:
        incremental = False
        train_data, train_dl = load_data(train_data, data_dir, incremental=incremental)
        val_data, val_dl = load_data(val_data, data_dir, incremental=incremental)
        test_data, test_dl = load_data(test_data, data_dir, incremental=incremental)

        n_classes = train_data.get_instance(0).num_classes
        print("Found {} classes".format(n_classes))

        if n_classes > 2: #onevsrest setup for more than 2 classes
            class_list = [str(i) for i in range(1,n_classes+1, 1)]
            for to_merge in combinations(class_list, n_classes-1):
                print("Merging classes ", to_merge)
                new_train_data = merge_classes(train_data, ','.join(to_merge))
                new_val_data = merge_classes(val_data, ','.join(to_merge))
                new_test_data = merge_classes(test_data, ','.join(to_merge))

                best_model = optimize_rule_params(classifier, new_train_data, new_val_data, new_test_data, incremental, train_dl, opt_metric) #merged attribute is always the last one, so 0 index for desired class
        else:
            best_model = optimize_rule_params(classifier, train_data, val_data, test_data, incremental, train_dl, opt_metric) #normal learning for binary cases

        save_model(best_model, out_fname, dir_out)

        idx_to_explain = 6837
        print("Finding the rule that covers instance {} of validation data".format(idx_to_explain))
        if classifier == 'jrip':
            get_rule_covering_inst(best_model, val_data, idx_to_explain)

    except Exception as e:
        print(e)
    finally:
        stop_jvm()


if __name__ == '__main__':
    classifier = 'jrip'
    optimize = 'macro-f-score'
    train_data = 'lstm_hid50_emb100_synthetic_min1_max6_skip1_train_pred.arff'
    val_data = 'lstm_hid50_emb100_synthetic_min1_max6_skip1_val_pred.arff'
    test_data = 'lstm_hid50_emb100_synthetic_min1_max6_skip1_test_pred.arff'
    data_dir = '../../out/weka/'

    out_fname = classifier +'_'+ train_data.strip('.arff') + '.model'
    dir_out = '../../out/weka/'
    induce_explanations(classifier, optimize, val_data, val_data, test_data, data_dir, out_fname, dir_out )