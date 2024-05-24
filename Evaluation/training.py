import pickle
import os
import numpy as np
from Join_scheme.data_prepare import process_stats_data, process_imdb_data
from Join_scheme.bound import Bound_ensemble
from Join_scheme.tools import get_n_bins_from_query
from BayesCard.Models.Bayescard_BN import Bayescard_BN
from BayesCard.Evaluation.cardinality_estimation import parse_query_single_table
from Sampling.create_binned_cols import create_binned_cols
from Sampling.get_query_binned_cards import get_query_binned_cards
from BayesCard.Testing.BN_training import train_stats
from time import perf_counter
import time
import matplotlib.pyplot as plt
import pandas as pd


def test_trained_BN_on_stats(bn, t_name):
    queries = {
        "posts": "SELECT COUNT(*) FROM posts as p WHERE posts.CommentCount<=18 AND posts.CreationDate>='2010-07-23 07:27:31'::timestamp AND posts.CreationDate<='2014-09-09 01:43:00'::timestamp",
        "comments": "SELECT COUNT(*) FROM comments as c WHERE comments.CreationDate>='2010-08-05 00:36:02'::timestamp AND comments.CreationDate<='2014-09-08 16:50:49'::timestamp",
        "postHistory": "SELECT COUNT(*) FROM postHistory as ph WHERE postHistory.PostHistoryTypeId=1 AND postHistory.CreationDate>='2010-09-14 11:59:07'::timestamp",
        "votes": "SELECT COUNT(*) FROM votes as v WHERE votes.VoteTypeId=2 AND votes.CreationDate<='2014-09-10 00:00:00'::timestamp",
        "postLinks": "SELECT COUNT(*) FROM postLinks as pl WHERE postLinks.LinkTypeId=1 AND postLinks.CreationDate>='2011-09-03 21:00:10'::timestamp AND postLinks.CreationDate<='2014-07-30 21:29:52'::timestamp",
        "users": "SELECT COUNT(*) FROM users as u WHERE users.DownVotes>=0 AND users.DownVotes<=0 AND users.UpVotes>=0 AND users.UpVotes<=31 AND users.CreationDate<='2014-08-06 20:38:52'::timestamp",
        "badges": "SELECT COUNT(*) FROM badges as b WHERE badges.Date>='2010-09-26 12:17:14'::timestamp",
        "tags": "SELECT COUNT(*) FROM tags"
    }

    true_cards = {
        "posts": 90764,
        "comments": 172156,
        "postHistory": 42308,
        "votes": 261476,
        "postLinks": 8776,
        "users": 37062,
        "badges": 77704,
        "tags": 1032
    }

    bn.init_inference_method()
    bn.infer_algo = "exact-jit"
    query = parse_query_single_table(queries[t_name], bn)
    pred = bn.query(query)
    print("pred is:", pred)
    assert min(pred, true_cards[t_name]) / max(pred, true_cards[t_name]) <= 1.5, f"Qerror too large, we have predition" \
                                                                                 f"{pred} for true card {true_cards[t_name]}"

    query = parse_query_single_table(queries[t_name], bn)
    _, id_probs = bn.query_id_prob(query, bn.id_attributes)
    if t_name not in ['votes', 'tags']:
        assert min(pred, np.sum(id_probs)) / max(pred, np.sum(id_probs)) <= 1.5, "query_id_prob is incorrect"


def test_trained_BN_on_stats_single():
    """
    train on single table sub-queries(card experiment)
    """
    # stats_CEB_single_table_sub_query_changed.sql
    query_location = ('/home/lrr/Documents/End-to-End-CardEst-Benchmark/workloads/stats_CEB/sub_plan_queries'
                      '/stats_CEB_single_table_sub_query.sql')
    model_path = '/home/lrr/Documents/research/FactorJoin/checkpoints/single_table_test/bin_200/'

    bn_dict = {}

    for file in os.listdir(model_path):
        table_file_path = os.path.join(model_path, file)
        table_name = file[:-4]
        # table_name = file[11: -4]
        print(table_name)
        with open(table_file_path, 'rb') as f:
            BN = pickle.load(f)

        if BN.infer_machine is None:
            BN.infer_algo = "exact-jit"
            BN.init_inference_method()
        bn_dict[table_name] = BN

    with open(query_location) as f:
        queries = f.readlines()

    # queries = {
    #     "posts": "SELECT COUNT(*) FROM posts as p WHERE posts.AnswerCount<=4 AND posts.CommentCount>=0 AND posts.CommentCount<=12 AND posts.FavoriteCount>=0 AND posts.FavoriteCount<=89 AND posts.CreationDate<='2014-09-02 10:21:04'::timestamp",
    #     "comments": "SELECT COUNT(*) FROM comments as c WHERE comments.CreationDate>='2010-08-05 00:36:02'::timestamp AND comments.CreationDate<='2014-09-08 16:50:49'::timestamp",
    #     "postHistory": "SELECT COUNT(*) FROM postHistory as ph WHERE postHistory.PostHistoryTypeId=1",
    #     "votes": "SELECT COUNT(*) FROM votes as v WHERE votes.VoteTypeId=2 AND votes.CreationDate<='2014-09-10 00:00:00'::timestamp",
    #     "postLinks": "SELECT COUNT(*) FROM postLinks as pl WHERE postLinks.LinkTypeId=1 AND postLinks.CreationDate>='2011-09-03 21:00:10'::timestamp AND postLinks.CreationDate<='2014-07-30 21:29:52'::timestamp",
    #     "users": "SELECT COUNT(*) FROM users as u WHERE users.DownVotes>=0 AND users.DownVotes<=0 AND users.UpVotes>=0 AND users.UpVotes<=31 AND users.CreationDate<='2014-08-06 20:38:52'::timestamp",
    #     "badges": "SELECT COUNT(*) FROM badges as b WHERE badges.Date>='2010-09-26 12:17:14'::timestamp",
    #     "tags": "SELECT COUNT(*) FROM tags"
    # }
    #
    # true_cards = {
    #     "posts": 12416,
    #     "comments": 172156,
    #     "postHistory": 42921,
    #     "votes": 261476,
    #     "postLinks": 8776,
    #     "users": 37062,
    #     "badges": 77704,
    #     "tags": 1032
    # }
    #
    # tables = ["posts",
    #           "comments",
    #           "postHistory",
    #           "votes",
    #           "postLinks",
    #           "users",
    #           "badges",
    #           "tags"]
    # for t_name in tables:
    #     bn = bn_dict[t_name]
    #     query = parse_query_single_table(queries[t_name], bn)
    #     pred = bn.query(query)
    #     print("pred is:", pred)
    #     print("true cardinality is:", true_cards[t_name])

    no_zero = 632
    predictions = []
    latencies = []
    q_errors = []
    ratios = []

    for query_no, query_str in enumerate(queries):
        cardinality_true = int(query_str.split("||")[-1])
        query_str = query_str.split("||")[0][:-1]
        table_name = query_str.split(" ")[3]

        # change alias to name
        query_str = change_alias_to_name(query_str, table_name)
        bn = bn_dict[table_name]
        bn.init_inference_method()
        bn.infer_algo = "exact-jit"

        print(f"Predicting cardinality for query {query_no}: {query_str}")
        query = parse_query_single_table(query_str, bn)
        # card_start_t = perf_counter()
        card_start_t = time.time()
        # cardinality_predict = BN.query(query, sample_size=sample_size)
        pred = float(bn.query(query))

        card_end_t = time.time()

        latency_ms = (card_end_t - card_start_t) * 1000
        # print("---prediction---", pred, "---true cardinality---", cardinality_true)
        if pred == 0 and cardinality_true == 0:
            q_error = 1.0
        elif np.isnan(pred) or pred == 0:
            pred = 1
            # q_error = max(pred / cardinality_true, cardinality_true / pred)
            q_error = pred / cardinality_true
        elif cardinality_true == 0:
            cardinality_true = 1
            # q_error = max(pred / cardinality_true, cardinality_true / pred)
            q_error = pred / cardinality_true
        else:
            # q_error = max(pred / cardinality_true, cardinality_true / pred)
            q_error = pred / cardinality_true

        if q_error > 2:
            print(f"latency: {latency_ms} and error: {q_error}")
        print(f"pred: {pred} and true_card: {cardinality_true}")
        predictions.append(pred)
        latencies.append(latency_ms)
        q_errors.append(q_error)
        ratios.append(q_error)

        # q_errors.append(q_error)
        # if cardinality_predict != 1:
        #     no_zero += 1
        #     q_errors.append(q_error)

    # save_predictions_to_file(
    #     predictions,
    #     latencies,
    #     "factorjoin",
    #     "factorjoin-time",
    #     "/home/lrr/Documents/research/card/results/stats/single_table/factorjoin_unchanged.csv",
    # )
    print("=====================================================================================")
    for i in [50, 90, 95, 99, 100]:
        print(f"q-error {i}% percentile is {np.percentile(q_errors, i)}")
    print(f"average latency is {np.mean(latencies)} ms")
    print(np.sum(latencies))
    # print(pre)
    # print(len(pre))
    # print(len(queries))
    print(no_zero)
    logbins = np.logspace(np.log10(min(ratios)), np.log10(max(ratios)), 100)
    plt.xscale("log")
    plt.hist(ratios, bins=logbins)
    plt.show()


def train_one_stats(dataset, data_path, model_folder, n_dim_dist=2, n_bins=200, bucket_method="greedy",
                    save_bucket_bins=False, seed=0, validate=True, actual_data=None):
    np.random.seed(seed)
    data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size = process_stats_data(data_path,
                                                                                                        model_folder,
                                                                                                        n_bins,
                                                                                                        bucket_method,
                                                                                                        save_bucket_bins,
                                                                                                        actual_data=actual_data)
    # Default value of n_bins is 200
    all_bns = dict()
    for table in schema.tables:
        t_name = table.table_name
        print(t_name)
        print(table.table_size)
        if t_name == "votes":
            print("null value!!!!!!!!!", null_values[t_name])
        bn = Bayescard_BN(t_name, key_attrs[t_name], bin_size[t_name], null_values=null_values[t_name])
        bn.build_from_data(data[t_name])
        if validate:
            test_trained_BN_on_stats(bn, t_name)
        all_bns[t_name] = bn

        model_path = model_folder + f"/{t_name}.pkl"
        pickle.dump(bn, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"model saved at {model_path}")

    # test_trained_BN_on_stats_single(all_bns)

    # multi-table training
    # be = Bound_ensemble(table_buckets, schema, n_dim_dist, bns=all_bns, null_value=null_values)
    #
    # if not os.path.exists(model_folder):
    #     os.mkdir(model_folder)
    # model_path = os.path.join(model_folder, f"model_{dataset}_{bucket_method}_{n_bins}.pkl")
    # pickle.dump(be, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    # print(f"models save at {model_path}")


def train_one_imdb(data_path, model_folder, n_dim_dist=1, bin_size=None, bucket_method="fixed_start_key",
                   sample_size=1000000, query_workload_file=None, save_bucket_bins=False, seed=0, prepare_sample=False,
                   db_conn_kwargs=None, sampling_percentage=1.0, sampling_type='ss', test_query_file=None,
                   materialize_sample=False):
    """
    Training one FactorJoin model on IMDB dataset.
    :param data_path: The path to IMDB dataset
    :param model_folder: The folder where we would like to save the trained models
    :param bin_size: The total number of bins we would like to assign to all keys.
           If set to None, we provide our hardcoded bin size derived by analyzing a similar workload to IMDB-JOB.
    :param query_workload_file: If there exists a query workload, we can use it to plan our binning budget.
    :param save_bucket_bins:
    :return:
    """
    np.random.seed(seed)
    if bin_size is None:
        # The following is determined by analyzing a similar workload to the IMDB-JOB workload
        n_bins = {
            'title.id': 800,
            'info_type.id': 100,
            'keyword.id': 100,
            'company_name.id': 100,
            'name.id': 100,
            'company_type.id': 100,
            'comp_cast_type.id': 50,
            'kind_type.id': 50,
            'char_name.id': 50,
            'role_type.id': 50,
            'link_type.id': 50
        }
    elif type(bin_size) == int:
        # need to provide a workload file to automatically generate number of bins
        n_bins = get_n_bins_from_query(bin_size, data_path, query_workload_file)
    else:
        assert type(bin_size) == dict, "bin_size must of type int or dictionary mapping id attributes to int"
        n_bins = bin_size

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    schema, table_buckets, ground_truth_factors_no_filter, bins, equivalent_keys = process_imdb_data(data_path,
                                                                                                     model_folder,
                                                                                                     n_bins,
                                                                                                     bucket_method,
                                                                                                     sample_size,
                                                                                                     save_bucket_bins,
                                                                                                     seed)
    be = Bound_ensemble(table_buckets, schema, n_dim_dist, ground_truth_factors_no_filter)
    if bin_size is None:
        bin_size = "default"
    elif type(bin_size) == dict:
        bin_size = "costumized"
    model_path = os.path.join(model_folder, f"model_imdb_{bin_size}.pkl")
    pickle.dump(be, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"models save at {model_path}")
    # create new tables for sampling purposes
    if prepare_sample:
        create_binned_cols(db_conn_kwargs, bins, equivalent_keys, sampling_percentage, sampling_type)
    if materialize_sample and test_query_file is not None:
        get_query_binned_cards(test_query_file, db_conn_kwargs, equivalent_keys, sampling_percentage, model_folder)


def train_one_stats_single_table(dataset, data_path, model_folder, n_dim_dist=2, n_bins=200, bucket_method="greedy",
                                 save_bucket_bins=False, seed=0, validate=True, actual_data=None):
    """
    Train single table model with BayesCard
    """
    # train_stats(data_path, model_folder, n_bins, save_bucket_bins)
    np.random.seed(seed)
    data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size = process_stats_data(data_path,
                                                                                                        model_folder,
                                                                                                        n_bins,
                                                                                                        bucket_method,
                                                                                                        save_bucket_bins,
                                                                                                        actual_data=actual_data)
    for table in schema.tables:
        t_name = table.table_name
        print(t_name)
        bn = Bayescard_BN(t_name, key_attrs[t_name], bin_size[t_name], null_values=null_values[t_name])
        bn.build_from_data(data[t_name])
        model_path = model_folder + f"/{t_name}.pkl"
        pickle.dump(bn, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"model saved at {model_path}")


def save_predictions_to_file(preds, times, header1, header2, file_path):
    data = zip(preds, times)
    df = pd.DataFrame(data, columns=[header1, header2])
    df.to_csv(file_path, index=False)
    print("----save done----")


def change_alias_to_name(query, table):
    tables = {
        "posts": ("posts.", "p."),
        "comments": ("comments.", "c."),
        "postHistory": ("postHistory.", "ph."),
        "votes": ("votes.", "v."),
        "postLinks": ("postLinks.", "pl."),
        "users": ("users.", "u."),
        "badges": ("badges.", "b."),
        "tags": ("tags.", "t.")
    }
    name = tables[table][0]
    alias = tables[table][1]
    query = query.replace(alias, name)
    return query


if __name__ == "__main__":
    test_trained_BN_on_stats_single()
    # queries = {
    #     "posts": "SELECT COUNT(*) FROM posts as p WHERE p.FavoriteCount>=0 AND p.CreationDate>='2010-07-23 02:00:12'::timestamp AND p.CreationDate<='2014-09-08 13:52:41'::timestamp",
    #     "comments": "SELECT COUNT(*) FROM comments as c WHERE c.Score=0",
    #
    # }
    # tables = ["posts", "comments"]
    # for i in range(2):
    #     table = tables[i]
    #     print(change_alias_to_name(queries[table], table))
