import os
import pickle
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import pingouin as pg
from import_qualtrics import get_qualtrics_survey

# ---------------------------------------------------- Result ------------------------------------------------------- #
sim = True
print_subjective = False
plot_subjective = False
plot_time = False
plot_weights = True

# plotting style
sns.set(style="darkgrid", context="talk")

# --------------------------------------------- Subjective response ------------------------------------------------- #

# download data from qualtrics
execution_survey_id = "SV_29ILBswADgbr79Q"
data_path = os.path.dirname(__file__) + "/data/"
# get_qualtrics_survey(dir_save_survey=data_path, survey_id=execution_survey_id)

# load user data
demo_path = data_path + "Human-Robot Assembly - Execution.csv"
df = pd.read_csv(demo_path)

# users to consider for evaluation
users = [8, 9, 10, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
user_idx = [df.index[df["Q0"] == str(user)][0] for user in users]
reactive_first_users = [8, 9, 10, 14, 15, 16, 17, 18]
proactive_first_users = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
reactive_first_user_idx = [df.index[df["Q0"] == str(user)][0] for user in reactive_first_users]
proactive_first_user_idx = [df.index[df["Q0"] == str(user)][0] for user in proactive_first_users]
condition1_q = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"]
condition2_q = ["Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30"]
plot_q = [[1]]
# plot_q = [[0, 1], [2, 3, 4, 5], [6, 7], [8, 9, 12], [10, 11]]

hxs, hys = [], []
for i, q_idx in enumerate(plot_q):
    X, Y, Z = [], [], []
    hx, hy = np.empty((18, 0)), np.empty((18, 0))
    for q_id in q_idx:
        c1q = condition1_q[q_id]
        c2q = condition2_q[q_id]

        x1 = df[c1q].iloc[reactive_first_user_idx]
        x2 = df[c2q].iloc[proactive_first_user_idx]

        y1 = df[c2q].iloc[reactive_first_user_idx]
        y2 = df[c1q].iloc[proactive_first_user_idx]

        x = np.concatenate([x1.to_numpy(dtype=float), x2.to_numpy(dtype=float)])
        y = np.concatenate([y1.to_numpy(dtype=float), y2.to_numpy(dtype=float)])

        # if c1q in ["Q3", "Q5", "Q10"]:
        #     x = (7 - x) + 1
        #     y = (7 - y) + 1

        # if c1q == "Q13":
        #     c1q = "Q11"

        # if c1q == "Q12":
        #     c1q = "Q13"
        #
        # if c1q == "Q11":
        #     c1q = "Q12"

        if print_subjective:
            np.savetxt('X.csv', x, delimiter=',')
            np.savetxt('Y.csv', y, delimiter=',')
            print(round(np.median(x), 3), round(np.median(y), 3))
            print("Wilcoxon:", stats.wilcoxon(x, y))
            print(" T-test :", stats.ttest_rel(x, y))

        X = X + (["reactive"] * len(x) + ["proactive"] * len(y))
        Y = Y + list(x) + list(y)
        Z = Z + [c1q] * (len(x)+len(y))

        hx = np.hstack([hx, np.reshape(x, (len(x), 1))])
        hy = np.hstack([hy, np.reshape(y, (len(y), 1))])

    # scale
    # print("Cronbach", pg.cronbach_alpha(pd.DataFrame(np.vstack([hx, hy]))))
    # print(round(np.mean(np.sum(hx, axis=1)), 3), round(stats.sem(np.sum(hx, axis=1)), 3),
    #       round(np.mean(np.sum(hy, axis=1)), 3), round(stats.sem(np.sum(hy, axis=1)), 3))
    # print("Pearson :", stats.pearsonr(np.sum(hx, axis=1), np.sum(hy, axis=1)))
    print(np.mean(hx), stats.sem(np.mean(hx, axis=1)), np.mean(hy), stats.sem(np.mean(hy, axis=1)))
    print(" T-test :", stats.ttest_rel(np.mean(hx, axis=1), np.mean(hy, axis=1)))

    hxs.append(np.sum(hx, axis=1))
    hys.append(np.sum(hy, axis=1))

    if plot_subjective:
        plt.figure(figsize=(2.5*len(q_idx), 5))
        g = sns.barplot(x=Z, y=Y, hue=X, ci=68, errwidth=2, capsize=.1, palette=["b", "g"])
        plt.ylim(1, 7)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel("User ratings", fontsize=22)
        # plt.legend([], [], frameon=False)
        plt.legend(fontsize=20, ncol=2, loc=1)
        plt.gcf().subplots_adjust(left=0.175)
        # plt.savefig("figures/corl/intelligence.png", bbox_inches='tight')
        plt.show()


# --------------------------------------------- Load accuracy data -------------------------------------------------- #

dir_path = os.path.dirname(__file__)
if sim:
    file_path = dir_path + "/results/corl_sim/"

    prior1_scores = np.loadtxt(file_path + "predict20_maxent_uni.csv")
    predict1_scores = np.loadtxt(file_path + "predict20_maxent_new_online2.csv")
    random1_scores = np.loadtxt(file_path + "random20_weights_new_online2.csv")

    prior2_scores = np.loadtxt(file_path + "predict20_maxent_uni_anti.csv")
    predict2_scores = np.loadtxt(file_path + "predict20_maxent_new_online_anti2.csv")
    random2_scores = np.loadtxt(file_path + "random20_weights_new_online_anti3.csv")

    # prior1_scores = np.loadtxt(file_path + "predict20_maxent_new_online_add0_all3.csv")
    # predict1_scores = np.loadtxt(file_path + "predict20_maxent_new_online_add0_all0.csv")
    # random1_scores = np.loadtxt(file_path + "random20_weights_new_online_add.csv")
    #
    # prior2_scores = np.loadtxt(file_path + "predict20_maxent_new_online_add1_all3.csv")
    # predict2_scores = np.loadtxt(file_path + "predict20_maxent_new_online_add1_all0.csv")
    # random2_scores = np.loadtxt(file_path + "random20_weights_new_online_noisy1.csv")

    prior3_scores = np.loadtxt(file_path + "predict20_maxent_new_online_add2_all3.csv")
    predict3_scores = np.loadtxt(file_path + "predict20_maxent_new_online_add2_all0.csv")
    random3_scores = np.loadtxt(file_path + "random20_weights_new_online_noisy2.csv")

    prior4_scores = np.loadtxt(file_path + "predict20_maxent_uni_noisy3.csv")
    predict4_scores = np.loadtxt(file_path + "predict20_maxent_new_online_noisy3.csv")
    random4_scores = np.loadtxt(file_path + "random20_weights_new_online_noisy3.csv")

    prior5_scores = np.loadtxt(file_path + "predict20_maxent_uni_noisy4.csv")
    predict5_scores = np.loadtxt(file_path + "predict20_maxent_new_online_noisy4.csv")
    random5_scores = np.loadtxt(file_path + "random20_weights_new_online_noisy4.csv")
else:
    # file_path = dir_path + "/results/corl/06-12/"
    # predict1_scores = np.loadtxt(file_path + "predict20_maxent_uni.csv")
    # predict2_scores = np.loadtxt(file_path + "predict20_maxent_uni_online_rand_new.csv")
    # random1_scores = np.loadtxt(file_path + "random20_weights_online_rand_new.csv")
    # random2_scores = np.loadtxt(file_path + "random20_actions.csv")
    file_path = dir_path + "/results/corl/"
    predict1_scores = np.loadtxt(file_path + "predict22_maxent_new_online.csv")
    predict2_scores = np.loadtxt(file_path + "predict22_maxent_new_online_add.csv")
    random1_scores = np.loadtxt(file_path + "random17_weights_new_online_add.csv")
    random2_scores = np.loadtxt(file_path + "predict17_maxent_uni.csv")

# ------------------------------------------------- Time taken ------------------------------------------------------ #

# plot result for user idle time
if plot_time:
    # compute result for user idle time
    times = pd.read_csv(file_path + "human18_task_times.csv", header=None)
    reactive_times = times[1]
    proactive_times = times[2]
    print("Reactive:", np.mean(reactive_times), stats.sem(reactive_times))
    print("Proactive:", np.mean(proactive_times), stats.sem(proactive_times))
    print("T-test:", stats.ttest_rel(reactive_times, proactive_times))

    x = list(reactive_times) + list(proactive_times)
    y = ["reactive "]*len(reactive_times) + ["proactive"]*len(proactive_times)
    plt.figure(figsize=(4.5, 5))
    # plt.figure()
    # sns.boxplot(y, x, width=0.7)
    sns.barplot(y, x, ci=68, errwidth=2, capsize=.1, palette=["b", "g"])
    plt.ylim(160, 190)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.ylabel("Time (s)", fontsize=18)
    plt.gcf().subplots_adjust(left=0.225)
    plt.show()
    # plt.savefig("figures/corl/task_time.png", bbox_inches='tight')


# ----------------------------------------------- Weights update ---------------------------------------------------- #

if plot_weights:
    true_weights = np.loadtxt(data_path + "user_demos/weights.csv").astype(float)
    learned_weights = pickle.load(open("results/corl_sim/learned_weights.csv", "rb"))
    updated_weights = pickle.load(open("results/corl_sim/updated_weights.csv", "rb"))
    updated_rand_weights = pickle.load(open("results/corl_sim/updated_rand_weights.csv", "rb"))

    # plot weights for given user
    ui = 1  # user id
    canonical_init = np.array(learned_weights[ui])[:, 0]
    canonical_learned = np.array(learned_weights[ui])[:, 1]
    complex_learned = np.array(updated_weights[ui])[:, -1]
    random_learned = np.array(updated_rand_weights[ui])[:, -1]
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(true_weights[ui][0], true_weights[ui][1], true_weights[ui][2], color="black", marker="*")
    ax.scatter3D(canonical_init[:, 0], canonical_init[:, 1], canonical_init[:, 2], color="red")
    ax.scatter3D(canonical_learned[:, 0], canonical_learned[:, 1], canonical_learned[:, 2], color="blue", marker="h")
    ax.scatter3D(complex_learned[:, 0], complex_learned[:, 1], complex_learned[:, 2], color="green", marker="^")
    ax.scatter3D(random_learned[:, 0], random_learned[:, 1], random_learned[:, 2], color="orange", marker="^")
    plt.show()

    # plot difference to weights
    transfer_weights_diff, random_weights_diff = [], []
    for ui, tw in enumerate(true_weights):
        transfer_diff = updated_weights[ui] - tw
        transfer_dist = np.linalg.norm(transfer_diff, axis=2)
        transfer_weights_diff.append(np.mean(transfer_dist, axis=0))

        random_diff = updated_rand_weights[ui] - tw
        random_dist = np.linalg.norm(random_diff, axis=2)
        random_weights_diff.append(np.mean(random_dist, axis=0))

    fig = plt.figure()
    y1 = np.mean(transfer_weights_diff, axis=0)
    y2 = np.mean(random_weights_diff, axis=0)
    x = range(len(y1))
    plt.plot(x, y1, "g-")
    plt.plot(x, y2, "r-")
    plt.show()

# --------------------------------------------- Action anticipation ------------------------------------------------- #
add = False
if not sim:
    if not add:
        # Split for adding feature
        # predict1_scores = [s for i, s in enumerate(predict1_scores) if i in [2, 5, 7, 10, 11, 12, 13, 14, 15, 17]]
        # predict2_scores = [s for i, s in enumerate(predict2_scores) if i in [2, 5, 7, 10, 11, 12, 13, 14, 15, 17]]
        # random1_scores = [s for i, s in enumerate(random1_scores) if i in [2, 5, 7, 10, 11, 12, 13, 14, 15, 17]]
        # random2_scores = [s for i, s in enumerate(random2_scores) if i in [2, 5, 7, 10, 11, 12, 13, 14, 15, 17]]

        # Split for comparing to random prior
        # predict1_scores = [s for i, s in enumerate(predict1_scores) if i in [2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18]]
        # predict2_scores = [s for i, s in enumerate(predict2_scores) if i in [2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18]]
        # random1_scores = [s for i, s in enumerate(random1_scores) if i in [2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18]]
        # random2_scores = [s for i, s in enumerate(random2_scores) if i in [2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18]]

        # All users
        predict1_scores = [s for i, s in enumerate(predict1_scores) if i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        predict2_scores = [s for i, s in enumerate(predict2_scores) if i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        random1_scores = [s for i, s in enumerate(random1_scores) if i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        random2_scores = [s for i, s in enumerate(random2_scores) if i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    # else:
    #     predict1_scores = predict1_scores[3:, :]
    #     predict2_scores = predict2_scores[3:, :]

n_users, n_steps = np.shape(predict1_scores)

# check statistical difference
predict1_users = list(np.sum(predict1_scores, axis=1)/n_steps)
predict2_users = list(np.sum(predict2_scores, axis=1)/n_steps)
random1_users = list(np.sum(random1_scores, axis=1)/n_steps)
random2_users = list(np.sum(random2_scores, axis=1)/n_steps)
if sim:
    predict3_users = list(np.sum(predict3_scores, axis=1) / n_steps)
    predict4_users = list(np.sum(predict4_scores, axis=1) / n_steps)
    predict5_users = list(np.sum(predict5_scores, axis=1) / n_steps)

    random3_users = list(np.sum(random3_scores, axis=1) / n_steps)
    random4_users = list(np.sum(random4_scores, axis=1) / n_steps)
    random5_users = list(np.sum(random5_scores, axis=1) / n_steps)

    prior1_users = list(np.sum(prior1_scores, axis=1) / n_steps)
    prior2_users = list(np.sum(prior2_scores, axis=1) / n_steps)
    prior3_users = list(np.sum(prior3_scores, axis=1) / n_steps)
    prior4_users = list(np.sum(prior4_scores, axis=1) / n_steps)
    prior5_users = list(np.sum(prior5_scores, axis=1) / n_steps)

print("predict 1:", predict1_users)
print("predict 2:", predict2_users)
print("random 1:", random1_users)
print("random 2:", random2_users)

if sim:
    print(" n-prior:", np.mean(predict1_users), stats.sem(predict1_users),
          np.mean(prior1_users), stats.sem(prior1_users),
          stats.ttest_rel(predict1_users, prior1_users))
    print("n-random:", np.mean(predict1_users), stats.sem(predict1_users),
          np.mean(random1_users), stats.sem(random1_users),
          stats.ttest_rel(predict1_users, random1_users))
    print(" ln-prior:", np.mean(predict2_users), stats.sem(predict2_users),
          np.mean(prior2_users), stats.sem(prior2_users),
          stats.ttest_rel(predict2_users, prior2_users))
    print("ln-random:", np.mean(predict2_users), stats.sem(predict2_users),
          np.mean(random2_users), stats.sem(random2_users),
          stats.ttest_rel(predict2_users, random2_users))
    print(" mn-prior:", np.mean(predict3_users), stats.sem(predict3_users),
          np.mean(prior3_users), stats.sem(prior3_users),
          stats.ttest_rel(predict3_users, prior3_users))
    print("mn-random:", np.mean(predict3_users), stats.sem(predict3_users),
          np.mean(random3_users), stats.sem(random3_users),
          stats.ttest_rel(predict3_users, random3_users))
else:
    # print("random:", np.mean(predict1_users), stats.sem(predict1_users),
    #       np.mean(random1_users), stats.sem(random1_users),
    #       stats.ttest_rel(predict1_users, random1_users))
    print("online:", np.mean(predict1_users), stats.sem(predict1_users),
          np.mean(predict2_users), stats.sem(predict2_users),
          stats.ttest_rel(predict1_users, predict2_users))
    # print(" prior:", np.mean(predict1_users), stats.sem(predict1_users),
    #       np.mean(random2_users), stats.sem(random2_users),
    #       stats.ttest_rel(predict1_users, random2_users))

plot_bar = True
cp = sns.color_palette()

if plot_bar:
    if sim:
        plt.figure(figsize=(8, 5))
        X = ["same"]*3*n_users + \
            ["opposite"]*3*n_users
        # X = ["add-phi1"] * 3 * n_users + \
        #     ["add-phi2"] * 3 * n_users
        Y = list(prior1_users) + list(random1_users) + list(predict1_users) + \
            list(prior2_users) + list(random2_users) + list(predict2_users)
        # Y = list(prior1_users) + list(predict1_users) + \
        #     list(prior2_users) + list(predict2_users) + \
        #     list(prior3_users) + list(predict3_users)
        Z = ["prior"]*n_users + ["rand_online"]*n_users + ["online"]*n_users + \
            ["prior"]*n_users + ["rand_online"]*n_users + ["online"]*n_users
        # Z = ["online"] * n_users + ["online_add"] * n_users + \
        #     ["online"] * n_users + ["online_add"] * n_users + \
        #     ["online"] * n_users + ["online_add"] * n_users
        sns.barplot(x=X, y=Y, hue=Z, ci=68, errwidth=2, capsize=.1)
        plt.ylim(0.55, 0.85)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=16)
        plt.ylabel("Accuracy", fontsize=20)
        plt.legend(fontsize=18, ncol=3, loc=9)
        # plt.gcf().subplots_adjust(bottom=0.175)
        # plt.gcf().subplots_adjust(left=0.175)
        # plt.savefig("figures/corl/sim_nop.png", bbox_inches='tight')
        plt.show()
    else:
        # plt.figure(figsize=(6, 5))
        # X = ["prior"] * n_users + ["rand_online"] * n_users + ["online"] * n_users
        # Y = list(predict1_users) + list(random1_users) + list(predict2_users)
        predict1_users = [12, 7, 13, 13, 14, 14, 11, 11, 11, 14]
        predict2_users = [15, 13, 14, 15, 14, 14, 15, 15, 14, 14]
        predict1_users = np.array(predict1_users) / 17.0
        predict2_users = np.array(predict2_users) / 17.0
        print(np.mean(predict1_users), stats.sem(predict1_users),
              np.mean(predict2_users), stats.sem(predict2_users),
              stats.ttest_rel(predict1_users, predict2_users))
        plt.figure(figsize=(5, 5))
        X = ["online"] * 10 + ["online_add"] * 10
        Y = list(predict1_users) + list(predict2_users)
        sns.barplot(x=X, y=Y, ci=68, errwidth=2, capsize=.1, palette=["g", "r", "b", cp[1]])
        # sns.boxplot(x=X, y=Y)  # ci=68, errwidth=2, capsize=.1)
        plt.ylim(0.45, 0.95)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=16)
        plt.ylabel("Accuracy", fontsize=18)
        # plt.legend(fontsize=18, ncol=2, loc=9)
        plt.gcf().subplots_adjust(bottom=0.175)
        plt.gcf().subplots_adjust(left=0.185)
        # plt.savefig("figures/corl/post_hoc.png", bbox_inches='tight')
        plt.show()
else:
    # accuracy over all users at each time step
    predict1_accuracy = np.sum(predict1_scores, axis=0) / n_users
    predict2_accuracy = np.sum(predict2_scores, axis=0) / n_users
    random1_accuracy = np.sum(random1_scores, axis=0) / n_users
    random2_accuracy = np.sum(random2_scores, axis=0) / n_users
    steps = np.array(range(len(predict1_accuracy))) + 1.0

    plt.figure(figsize=(9, 5))
    plt.plot(steps, random2_accuracy, 'r:', linewidth=4.5, alpha=0.95)
    plt.plot(steps, random1_accuracy, 'y-.', linewidth=4.5, alpha=0.95)
    plt.plot(steps, predict2_accuracy, 'b--', linewidth=4.5, alpha=0.95)
    plt.plot(steps, predict1_accuracy, 'g', linewidth=4.5, alpha=0.95)
    plt.xlim(0, 18)
    plt.ylim(-0.1, 1.1)
    plt.xticks(steps, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Time step", fontsize=22)
    plt.ylabel("Accuracy", fontsize=22)
    # plt.title("Action prediction using personalized priors", fontsize=22)
    plt.gcf().subplots_adjust(bottom=0.175)
    plt.legend(["random action", "random weights", "initial estimate", "removed feat"], fontsize=20, ncol=2, loc=8)
    plt.show()
    # plt.savefig("figures/corl/online_accuracy.png", bbox_inches='tight')
