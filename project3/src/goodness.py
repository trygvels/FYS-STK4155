import time
import numpy as np
import random

class Goodness:  # Logistic regression class
    def __init__(self):
        self.print_goodness = True

    def classification_report(
        self,
        ytrue,
        pred,
        print_to_file=False,
        print_to_latex_file=False,
        print_res=True,
        return_ac=False,
        return_f1=False,
        return_f1_weight=False,
        return_ar=False,
        return_ar_weight=False,
        return_cp=False,
        debug=False,
        label=None,
    ):
        """
        Function to calculate and print the goodness of fit parameters for a given 
        prediction.
        Input: numpy arrays of size (N_data, N_classes)
               ytrue: array of true classification {0,1} 
               pred:  array of predicted classification, probabilities [0,1]
        The images (row) is predicted to belong to the class (column) with the highes prob.
        """
        #####################
        """
        Code to compute accurracy and F1-score
        code finished and tested
        """

        n = pred.shape[0]
        m = pred.shape[1]
        tp = np.zeros(shape=(m), dtype="int")  # number of true positive predictions per class
        tn = np.zeros(shape=(m), dtype="int")  # number of true negative predictions per class
        fp = np.zeros(shape=(m), dtype="int")  # number of false positive predictions per class
        fn = np.zeros(shape=(m), dtype="int")  # number of false negative predictions per class
        pred_bin = np.zeros(shape=(n, m), dtype="int")  # binary array of predictions

        # set class prediction and count true/flase positives/negatives
        for i in range(n):
            ind = np.argmax(pred[i, :])  # index of higher probability of row i
            pred_bin[i, ind] = 1  # set (row i,column ind) to 1 (rest is zero)
            for j in range(m):  # this may be vectorized
                if pred_bin[i, j] == 1 and ytrue[i, j] == 1:
                    tp[j] += 1
                elif pred_bin[i, j] == 1 and ytrue[i, j] == 0:
                    fp[j] += 1
                elif pred_bin[i, j] == 0 and ytrue[i, j] == 0:
                    tn[j] += 1
                elif pred_bin[i, j] == 0 and ytrue[i, j] == 1:
                    fn[j] += 1

        # count number of predicted targets and true targets per class
        pcp = np.sum(np.where(pred_bin == 1, 1, 0), axis=0)  # predicted number of targets per class
        cp = np.sum(np.where(ytrue == 1, 1, 0), axis=0)  # true number of targets per class

        # the following code is written such as to not divide by zero.
        # calculate positive predictive value (precission) and true positive rate (recall)
        ppv = tp * 1.0
        tpr = tp * 1.0
        for i in range(len(tp)):
            if tp[i] == 0:
                ppv[i] = 0.0
                tpr[i] = 0.0
            else:
                ppv[i] = tp[i] * 1.0 / pcp[i]
                tpr[i] = tp[i] * 1.0 / cp[i]

        # calculate accuracy and F1-score (per calss) and weighted F1-score
        ac = (
            np.sum(tp) * 1.0 / n
        )  # accuracy: sum all correct predictions (positive) and divide by total number of images (rows)
        f1 = ppv.copy()
        for i in range(len(tp)):
            if ppv[i] == 0.0 and tpr[i] == 0.0:
                f1[i] = 0.0
            else:
                f1[i] = 2.0 * ppv[i] * tpr[i] / (ppv[i] + tpr[i])  # f1 score per class

        f1_weight = (
            np.sum(f1 * cp) / n
        )  # weight each f1 score with the relative number of images per class (the true values, not the predicted)

        # temporary print
        if debug:
            print("ac", ac)
            print("f1_weight", f1_weight)
            print("f1", f1)
            print("cp", cp)
            print()

        #############################
        """
        Code to calculate area ratio/ area under the curve per class
        code finished and tested
        """
        ar = np.zeros(m)
        auc = ar.copy()

        # run over each class
        for j in range(m):
            # sort probabilities of each class from highest to lowest
            """
            ind_arr=np.arange(0,n,dtype='int') #array keeping track of unsorted indices
            """
            sorted_ind = np.zeros(shape=n, dtype="int")  # array keeping track of sorted indices
            temp_pred = pred[:, j].copy()  # array keeping track of unsorted probabilities
            sorted_pred = np.zeros(n)  # sorted probabilities (not used, only for debug)

            t1 = time.time()
            # sort the probabilities from highest to lowest
            for i in range(n):
                ind = np.argmax(temp_pred)  # find index to maximum of temp_pred
                sorted_ind[i] = ind  # assign correct index to sorted index array
                temp_pred[ind] = -1.0  # set temp probability to negative value
                # this is lower than any true probability
                # We still search on whole array, but reducing
                # the array will be equivalently expencive and
                # require more code
            t2 = time.time()
            if debug:  # debug
                print("total sorting time [s]:", t2 - t1)

            # compute the area ratio (AR)
            ycum_ar = np.zeros(n + 1, dtype="int")  # cumulative curve for AR
            for i in range(n):
                ycum_ar[i + 1] = ycum_ar[i] + ytrue[sorted_ind[i], j]

            # trapezoid integration
            area_under_model = np.sum(ycum_ar) * 1.0 - 0.5 * (ycum_ar[0] + ycum_ar[-1])
            area_under_baseline = cp[j] * n * 0.5
            area_best_fit = 0.5 * cp[j] ** 2 + 1.0 * cp[j] * (n - cp[j])
            base_area = area_best_fit - area_under_baseline
            model_area = area_under_model - area_under_baseline
            ar[j] = model_area / base_area

            # compute the area under the curve (AUC)
            ycum_auc = np.zeros(n - cp[j], dtype="int")  # cumulative curve for AUC
            ind_auc = 0
            for i in range(n):
                if ytrue[sorted_ind[i], j] == 1:
                    ycum_auc[ind_auc] += 1
                else:
                    ind_auc += 1
                    if ind_auc == ycum_auc.shape[0]:  # reached end
                        break
                    ycum_auc[ind_auc] = ycum_auc[ind_auc - 1]

            area_under_model = np.sum(ycum_auc) * 1.0
            total_area = cp[j] * (n - cp[j]) * 1.0
            auc[j] = area_under_model / total_area

            if debug:  # debug
                print("class", j)
                print("ar", ar[j], "auc", auc[j])
                print("auc->ar", 2 * (auc[j] - 0.5), "ar->auc", 0.5 + ar[j] / 2.0)
                print()

        ar_weight = np.sum(ar * cp) / n
        auc_weight = np.sum(auc * cp) / n

        # print a good and easily viewable classification
        """
        TP:  number of true positives (correctly predicted targets)
        PCP: number of predicted positives (targets)
        CP:  number of positives (targets)
        PPV: Positive predictive value (precision)
        TPR: True positive rate (recall)
        class    TP    PCP     CP    PPV    TPR   F1-score     AR      AUC 
        ---------------------------------------------------------------------
          0      123    163    155  .....  .....   .......  .......  .......
          1    .....  .....  .....  .....  .....   .......  .......  .......
          2    .....  .....  .....  .....  .....   .......  .......  .......
          3    .....  .....  .....  .....  .....   .......  .......  .......
          4    .....  .....  .....  .....  .....   .......  .......  .......
          5    .....  .....  .....  .....  .....   .......  .......  .......
          6    .....  .....  .....  .....  .....   .......  .......  .......
          7    .....  .....  .....  .....  .....   .......  .......  .......
          8    .....  .....  .....  .....  .....   .......  .......  .......
        ...    .....  .....  .....  .....  .....   .......  .......  .......
        ...    .....  .....  .....  .....  .....   .......  .......  .......
        ...    .....  .....  .....  .....  .....   .......  .......  .......
                             CP weighted average   .......  .......  .......
                                  Total accuracy   .......
        ---------------------------------------------------------------------
        """
        if print_res:
            print("class    TP    PCP     CP    PPV    TPR   F1-score     AR      AUC ")
            print("---------------------------------------------------------------------")
            for j in range(m):
                print(
                    "%3i    %5i  %5i  %5i  %5.3f  %5.3f   %7.4f  %7.4f  %7.4f"
                    % (j, tp[j], pcp[j], cp[j], ppv[j], tpr[j], f1[j], ar[j], auc[j])
                )
            print()
            print("                     CP weighted average   %7.4f  %7.4f  %7.4f" % (f1_weight, ar_weight, auc_weight))
            print()
            print("                          Total accuracy   %7.4f" % (ac))
            print("---------------------------------------------------------------------")
            print("")

        # print table values to file (to save them)
        if print_to_file:
            t = time.ctime()
            ta = t.split()
            hms = ta[3].split(":")
            lab = ta[4] + "_" + ta[1] + ta[2] + "_" + hms[0] + hms[1] + hms[2]
            if label == None:
                label = lab
            else:
                label = label + "_" + lab
            filename = "../goodness_parameters/parameters_goodness_" + label + ".txt"
            out = open(filename, "w")
            out.write("class    TP    PCP     CP    PPV    TPR   F1-score     AR      AUC \n")
            out.write("---------------------------------------------------------------------\n")
            for j in range(m):
                out.write(
                    "%3i    %5i  %5i  %5i  %5.3f  %5.3f   %7.4f  %7.4f  %7.4f\n"
                    % (j, tp[j], pcp[j], cp[j], ppv[j], tpr[j], f1[j], ar[j], auc[j])
                )
            out.write("\n")
            out.write(
                "                     CP weighted average   %7.4f  %7.4f  %7.4f\n" % (f1_weight, ar_weight, auc_weight)
            )
            out.write("\n")
            out.write("                          Total accuracy   %7.4f\n" % (ac))
            out.write("---------------------------------------------------------------------\n")
            out.close()



        # print table values for latex tables to file (to save them)
        if print_to_latex_file:
            t = time.ctime()
            ta = t.split()
            hms = ta[3].split(":")
            lab = 'latex_' + ta[4] + "_" + ta[1] + ta[2] + "_" + hms[0] + hms[1]
            if label == None:
                label = lab
            else:
                label = label + "_" + lab
            filename = "../goodness_parameters/parameters_goodness_" + label + ".txt"
            out = open(filename, "w")
            out.write("class    F1-score     AR      AUC \n")
            out.write("---------------------------------------------------------------------\n")
            for j in range(m):
                out.write(
                    "%3i  &  %7.4f &  %7.4f &  %7.4f \n"
                    % (j,  f1[j], ar[j], auc[j])
                )
            out.write("\n")
            out.write(
                "                     CP weighted average &  %7.4f & %7.4f &  %7.4f\n" % (f1_weight, ar_weight, auc_weight)
            )
            out.write("---------------------------------------------------------------------\n")
            out.close()

        # return the desired values
        """
        Tested and is working:
        Note: all return argumants are on numpy array format
        order of output:
        accuracy, f1, f1_weight, ar, ar_weight 
        """
        narg = 0
        arg_ind = 0
        arg_vec = np.zeros(3 + m * 3 + 1)
        arg_split = np.zeros(6, dtype="int")

        if debug:
            print(ac)
            print(f1)
            print(f1_weight)
            print(ar)
            print(ar_weight)

        # Find which arguments to return and add to argument array
        if return_ac:
            arg_vec[arg_ind] = ac
            arg_split[narg] = arg_ind + 1
            narg += 1
            arg_ind += 1
        if return_f1:
            arg_vec[arg_ind : arg_ind + m] = f1.copy()
            arg_split[narg] = arg_ind + m
            narg += 1
            arg_ind += m
        if return_f1_weight:
            arg_vec[arg_ind] = f1_weight
            arg_split[narg] = arg_ind + 1
            narg += 1
            arg_ind += 1
        if return_ar:
            arg_vec[arg_ind : arg_ind + m] = ar.copy()
            arg_split[narg] = arg_ind + m
            narg += 1
            arg_ind += m
        if return_ar_weight:
            arg_vec[arg_ind] = ar_weight
            arg_split[narg] = arg_ind + 1
            narg += 1
            arg_ind += 1
        if return_cp:
            cp=cp*1.0
            arg_vec[arg_ind : arg_ind + m] = cp.copy()
            arg_split[narg] = arg_ind + m
            narg += 1
            arg_ind += m

        # depending on number of outputs, split argument array and return the given
        # number of arguments
        if narg == 1:
            a0 = arg_vec[0 : arg_split[0]]
            return a0
        elif narg == 2:
            a0 = arg_vec[: arg_split[0]]
            a1 = arg_vec[arg_split[0] : arg_split[1]]
            return a0, a1
        elif narg == 3:
            a0 = arg_vec[: arg_split[0]]
            a1 = arg_vec[arg_split[0] : arg_split[1]]
            a2 = arg_vec[arg_split[1] : arg_split[2]]
            return a0, a1, a2
        elif narg == 4:
            a0 = arg_vec[: arg_split[0]]
            a1 = arg_vec[arg_split[0] : arg_split[1]]
            a2 = arg_vec[arg_split[1] : arg_split[2]]
            a3 = arg_vec[arg_split[2] : arg_split[3]]
            return a0, a1, a2, a3
        elif narg == 5:
            a0 = arg_vec[: arg_split[0]]
            a1 = arg_vec[arg_split[0] : arg_split[1]]
            a2 = arg_vec[arg_split[1] : arg_split[2]]
            a3 = arg_vec[arg_split[2] : arg_split[3]]
            a4 = arg_vec[arg_split[3] : arg_split[4]]
            return a0, a1, a2, a3, a4
        elif narg == 6:
            a0 = arg_vec[: arg_split[0]]
            a1 = arg_vec[arg_split[0] : arg_split[1]]
            a2 = arg_vec[arg_split[1] : arg_split[2]]
            a3 = arg_vec[arg_split[2] : arg_split[3]]
            a4 = arg_vec[arg_split[3] : arg_split[4]]
            a5 = arg_vec[arg_split[4] : arg_split[5]]
            return a0, a1, a2, a3, a4, a5
        else:
            return  # no arguments

    def test_classification(self, debug=False):
        # Write a short code to test the above
        """
        do random values of an 20000x10 array
        normalize each row
        create random (ish) ytrue array (20% chance of not recalling highest probability, but a random one)
        """
        np.random.seed(235253)
        ypred = np.random.uniform(0, 1, size=(20000, 10))
        ytrue = np.zeros(shape=(ypred.shape), dtype="int")
        for i in range(ypred.shape[0]):
            ypred[i, :] = ypred[i, :] / np.sum(ypred[i, :])
            ind = np.argmax(ypred[i, :])
            r1 = np.random.uniform(0, 1)
            if r1 > 0.80:  # in 80% of the cases ytrue is set by argmax, else random class
                ind = np.random.randint(0, ypred.shape[1])
            ytrue[i, ind] = 1

        if debug:
            ac = -1
            f1 = -1
            f1w = -1
            ar = -1
            arw = -1

            ac, f1, f1w, ar, arw = self.classification_report(
                ytrue,
                ypred,
                print_to_file=False,
                print_res=True,
                return_ac=True,
                return_f1=True,
                return_f1_weight=True,
                return_ar=True,
                return_ar_weight=True,
                debug=debug,
            )
        else:
            self.classification_report(
                ytrue,
                ypred,
                print_to_file=False,
                print_to_latex_file=True,
                label="test",
                print_res=True,
                return_ac=False,
                return_f1=False,
                return_f1_weight=False,
                return_ar=False,
                return_ar_weight=False,
                debug=debug,
            )

        if debug:
            print()
            print(ac)
            print(f1)
            print(f1w)
            print(ar)
            print(arw)
