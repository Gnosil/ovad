#!/usr/bin/env python3

# Copyright (c)  2021  Johns Hopkins University (authors: Desh Raj)
# Apache 2.0
import argparse
from itertools import groupby
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomHdf5Writer, SupervisionSegment
from lhotse.manipulation import combine
from lhotse.recipes import (prepare_ami, prepare_cslu_kids, 
prepare_gale_arabic, prepare_gale_mandarin, prepare_librispeech, prepare_nsc, 
prepare_cmu_kids, prepare_mtedx, prepare_switchboard, prepare_tedlium)


# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@contextmanager
def get_executor():
    # We'll either return a process pool or a distributed worker pool.
    # Note that this has to be a context manager because we might use multiple
    # context manager ("with" clauses) inside, and this way everything will
    # free up the resources at the right time.
    try:
        # If this is executed on the CLSP grid, we will try to use the
        # Grid Engine to distribute the tasks.
        # Other clusters can also benefit from that, provided a cluster-specific wrapper.
        # (see https://github.com/pzelasko/plz for reference)
        #
        # The following must be installed:
        # $ pip install dask distributed
        # $ pip install git+https://github.com/pzelasko/plz
        name = subprocess.check_output("hostname -f", shell=True, text=True)
        if name.strip().endswith(".clsp.jhu.edu"):
            import plz
            from distributed import Client

            with plz.setup_cluster(memory="6G") as cluster:
                cluster.scale(80)
                yield Client(cluster)
            return
    except:
        pass
    # No need to return anything - compute_and_store_features
    # will just instantiate the pool itself.
    yield None


def locate_corpus(*corpus_dirs):
    for d in corpus_dirs:
        if os.path.exists(d):
            return d
    print(
        "Please create a place on your system to put the downloaded Librispeech data "
        "and add it to `corpus_dirs`"
    )
    sys.exit(1)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num-jobs", type=int, default=min(15, os.cpu_count()))
    return parser

def supervision_alignment(SupervisionSet,supervision_file,ctm):
    return SupervisionSet.from_json(
        supervision_file
    ).with_alignment_from_ctm(ctm)


def main():
    args = get_parser().parse_args()

    ami_corpus_dir = Path("/export/corpora5/amicorpus")
    cslu_kids_dir = Path("/export/corpora5/LDC/LDC2007S18")
    cmu_kids_dir = Path("/export/corpora5/LDC/LDC97S63")

    gale_arabic_audio_dir = [Path("/export/corpora5/LDC/LDC2013S02")]

    gale_arabic_text_dir = [Path("/export/corpora5/LDC/LDC2013T17")]

    gale_mandarin_audio_dir = [Path("/export/corpora5/LDC/LDC2013S08"),
    Path("/export/corpora5/LDC/LDC2013S04"),
    Path("/export/corpora5/LDC/LDC2014S09"),
    Path("/export/corpora5/LDC/LDC2015S06"),
    Path("/export/corpora5/LDC/LDC2015S13"),
    Path("/export/corpora5/LDC/LDC2016S03")]

    gale_mandarin_text_dir = [Path("/export/corpora5/LDC/LDC2013T20"),
    Path("/export/corpora5/LDC/LDC2013T08"),
    Path("/export/corpora5/LDC/LDC2014T28"),
    Path("/export/corpora5/LDC/LDC2015T09"),
    # Path("/export/corpora5/LDC/LDC2015T25"),
    Path("/export/corpora5/LDC/LDC2016T12")]

    librispeech_dir = Path("/export/corpora5/LibriSpeech")
    nsc_dir = Path("/export/corpora5/nsc")
    mtedx_dir = Path("/export/corpora5/mTEDx")
    switchboard_dir = Path("")
    tedlium_dir = Path("/export/corpora5/TEDLIUM_release-3")
    
    annotations_dir = Path("/export/c07/sli218")

    # download_ami(corpus_dir, annotations=annotations_dir, mic="sdm")

    output_dir = Path("exp/data")

    ami_ctm_dir = "/export/c07/sli218/kaldi/egs/ami/s5b/ami_ctm/train.ctm"
    cmu_cslu_ctm_dir = Path("/export/c07/sli218/kaldi/egs/cmu_cslu_kids/s5/cmu_cslu_ctm")
    gale_arabic_ctm_dir = Path("/export/c07/sli218/kaldi/egs/gale_arabic/s5d/gale_arabic_ctm")
    gale_arabic_mandarin_dir = Path("/export/c07/sli218/kaldi/egs/gale_mandarin/s5/gale_mandarin_ctm")
    librispeech_ctm_dir = Path("/export/c07/sli218/kaldi/egs/librispeech/s5/libri_ctm")
    mtedx_ctm_dir = Path("/home/mwiesner/tedx.noarabic.ctm")
    nsc_ctm_dir = Path("/export/c07/sli218/kaldi/egs/nsc/s5/nsc_ctm")
    switchboard_ctm_dir = Path("/export/c07/sli218/kaldi/egs/swbd/s5c/swbd_ctm")
    tedlium_ctm_dir = Path("/export/c07/sli218/kaldi/egs/tedlium/s5_r3/tedlium_ctm")
    # cmu_kids_ctm_dir = Path("/home/mwiesner/tedx.noarabic.ctm")


    # with open(f'/export/c07/sli218/kaldi/egs/ami/s5b/ami_ctm/train.ctm', 'r') as f_out:
    #     text=""
    #     for line in f_out:
    #         content = line.strip().split()[:5]
    #         if content[1]=="A":
    #             content[1]="1"
    #         text = text+" ".join(content)+"\n"

    # with open(f'/export/c07/sli218/kaldi/egs/ami/s5b/ami_ctm/train.ctm', 'w') as f_out:
    #     f_out.write(text)
    # # Replace string
    # content = content.replace("A", "1")
    # # Write new content in write mode 'w'
    # with open(f'/export/c07/sli218/kaldi/egs/ami/s5b/ami_ctm/dev.ctm', 'w') as file:
    #     file.write(content)

    print("AMI manifest preparation:")
    ami_manifests = prepare_ami(
        ami_corpus_dir,
        # annotations_dir=annotations_dir,
        output_dir=output_dir,
        mic="sdm",
        partition="full-corpus",
        # max_pause=0,
    )
    print(ami_manifests)
    cuts_ami = CutSet.from_manifests(
        **ami_manifests["train"]
    ).trim_to_supervisions()
    # # ami_manifests["train"]["supervisions"].with_alignment_from_ctm(ami_ctm_dir)

    # print("CSLU kids manifest preparation:")
    # cslu_kids_manifests = prepare_cslu_kids(
    #     cslu_kids_dir,
    #     # annotations_dir=annotations_dir,
    #     output_dir=output_dir,
    #     # mic="sdm",
    #     # partition="full-corpus",
    #     # max_pause=0,
    # )
    # print(cslu_kids_manifests)
    # cuts_cslu_kids = CutSet.from_manifests(
    #     **cslu_kids_manifests
    # ).trim_to_supervisions()

    # print("gale arabic manifest preparation:")
    # gale_arabic_manifests = prepare_gale_arabic(
    #     gale_arabic_audio_dir,
    #     gale_arabic_text_dir,
    #     # annotations_dir=annotations_dir,
    #     output_dir=output_dir
    # )
    # cuts_gale_arabic = CutSet.from_manifests(
    #     **gale_arabic_manifests["train"]
    # ).trim_to_supervisions()

    # print("gale manderin manifest preparation:")
    # gale_mandarin_manifests = prepare_gale_mandarin(
    #     gale_mandarin_audio_dir,
    #     gale_mandarin_text_dir,
    #     # annotations_dir=annotations_dir,
    #     output_dir=output_dir
    #     # mic="sdm",
    #     # partition="full-corpus",
    #     # max_pause=0,
    # )
    # cuts_gale_mandarin = CutSet.from_manifests(
    #     **gale_mandarin_manifests["train"]
    # ).trim_to_supervisions()

    # print("LibriSpeech manifest preparation:")
    # librispeech_manifests = prepare_librispeech(
    #     librispeech_dir,
    #     dataset_parts = "mini-librispeech",
    #     # annotations_dir=annotations_dir,
    #     output_dir=output_dir,
    #     # mic="sdm",
    #     # partition="full-corpus",
    #     # max_pause=0,
    # )
    

    # print("NSC manifest preparation:")
    # nsc_manifests = prepare_nsc(
    #     nsc_dir,
    #     # annotations_dir=annotations_dir,
    #     output_dir=output_dir,
    #     # mic="sdm",
    #     # partition="full-corpus",
    #     # max_pause=0,
    # )
    # cuts_nsc = CutSet.from_manifests(
    #     **nsc_manifests["train"]
    # ).trim_to_supervisions()

    # print("mTEDx manifest preparation:")
    # mtedx_manifests = prepare_mtedx(
    #     mtedx_dir,
    #     # annotations_dir=annotations_dir,
    #     output_dir=output_dir,
    #     # mic="sdm",
    #     # partition="full-corpus",
    #     # max_pause=0,
    # )
    # cuts_mtedx = CutSet.from_manifests(
    #     **mtedx_manifests["train"]
    # ).trim_to_supervisions()

    # print("Switchboard manifest preparation:")
    # switchboard_manifests = prepare_switchboard(
    #     switchboard_dir,
    #     # annotations_dir=annotations_dir,
    #     output_dir=output_dir,
    #     # mic="sdm",
    #     # partition="full-corpus",
    #     # max_pause=0,
    # )
    # cuts_switchboard = CutSet.from_manifests(
    #     **switchboard_manifests["train"]
    # ).trim_to_supervisions()

    # print("TEDlium manifest preparation:")
    # tedlium_manifests = prepare_tedlium(
    #     tedlium_dir,
    #     # annotations_dir=annotations_dir,
    #     output_dir=output_dir,
    #     # mic="sdm",
    #     # partition="full-corpus",
    #     # max_pause=0,
    # )
    # cuts_tedlium = CutSet.from_manifests(
    #     **tedlium_manifests["train"]
    # ).trim_to_supervisions()
    # storage_path = output_dir

    # manifests = [ami_manifests,
    #     librispeech_manifests,
    #     mtedx_manifests]
    # manifests = sorted(manifests, key=lambda cut: cut.features.storage_path)
    # subsets = groupby(manifests, lambda cut: cut.features.storage_path)
    # unique_storage_paths, subsets = zip(
    #         *[(k, CutSet.from_cuts(grp)) for k, grp in subsets]
    #     )
    
    # # Create paths for new feature files and subset cutsets.
    # tot_items = len(unique_storage_paths)
    # new_storage_paths = [f"{storage_path}/feats-{i}" for i in range(tot_items)]
    # partial_manifest_paths = [
    #     f"{storage_path}/cuts-{i}.jsonl.gz" for i in range(tot_items)
    # ]

    # num_jobs = len(unique_storage_paths)
    
    # # Create directory if needed (storage_path might be an URL)
    # if Path(storage_path).parent.is_dir():
    #     Path(storage_path).mkdir(exist_ok=True)
    # with ProcessPoolExecutor(num_jobs) as ex:
    #         futures = []
    #         for cs, nsp, pmp in zip(subsets, new_storage_paths, partial_manifest_paths):
    #             futures.append(ex.submit(copy_feats_worker, cs, nsp, storage_type, pmp))

    #         all_cuts = combine_manifests((f.result() for f in as_completed(futures)))
    print(ami_manifests)
    # print(librispeech_manifests)
    # print(mtedx_manifests)
    # onlyfiles = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)) and f.endswith(".gz")]
    # universe_manifests = combine(
    #     onlyfiles
    #     # gale_manderain_manifests
    #     )
    #     cuts_cslu_kids,
    #     cuts_gale_arabic, 
    #     cuts_gale_mandarin,
    #     cuts_librispeech,
    #     cuts_nsc,
    #     cuts_mtedx,
    #     cuts_switchboard,
    #     cuts_tedlium)

    # print(universe_manifests)
    print("Feature extraction:")
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    with get_executor() as ex:  # Initialize the executor only once.
        for partition, manifests in ami_manifests.items():
            ''' print(manifests["supervisions"])
            manifests["supervisions"]=manifests["supervisions"].from_jsonl("exp/data/supervisions_dev.jsonl").with_alignment_from_ctm(
                    ctm_dir,
            ) '''
            if (output_dir / f"ami_cuts_{partition}.jsonl.gz").is_file():
                print(f"{partition} already exists - skipping.")
                continue
            print("Processing", partition)
            print(manifests.keys())
            cut_set = CutSet.from_manifests(
                recordings=manifests["recordings"],
                supervisions=manifests["supervisions"],
            ).cut_into_windows(duration=5)
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=args.num_jobs if ex is None else min(80, len(cut_set)),
                executor=ex,
                storage_type=LilcomHdf5Writer,
            ).pad(duration=5.0)
            cut_set.to_jsonl(output_dir / f"ami_cuts_{partition}.jsonl.gz")


if __name__ == "__main__":
    main()
