import json
import logging
import os

from argparse import Namespace

import numpy as np

from fairseq import metrics, options, utils
from fairseq.data import data_utils, encoders
from fairseq.data.sign_dataset import SignDataset
from fairseq.data.sign_language_pair_dataset import SignLanguagePairDataset
from fairseq.tasks import register_task, FairseqTask

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4


def load_sign_dataset(filepath, src_feat_roots):
    with open(filepath, 'r') as ipf:
        content = json.load(ipf)

    identifiers = [c['ident'] for c in content]
    sizes = [c['size'] for c in content]

    return SignDataset(identifiers, sizes, src_feat_roots)


def load_langpair_dataset(
    data_path, split,
    src, src_feat_roots,
    tgt, tgt_dict,
    dataset_impl,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, multilv_args, prepend_bos=False, load_alignments=False,
    truncate_source=False, use_bucketing=True
):

    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))

    src_dataset = load_sign_dataset(prefix + src, src_feat_roots)
    tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)

    assert len(src_dataset) == len(tgt_dataset)

    logger.info('{} {} {}-{} {} examples'.format(
        data_path, split, src, tgt, len(src_dataset)
    ))

    # if prepend_bos:
    #     assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
    #     src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
    #     tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None
    return SignLanguagePairDataset(
        src_dataset, src_dataset.sizes,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset,
        use_bucketing=use_bucketing,
        multilv_args=multilv_args
    )


@register_task('translation_sign')
class TranslationSign(FairseqTask):
    """
    Translation from sign videos into spoken language.
    
    """

    @staticmethod
    def add_args(parser):
        """
        Add task-specific arguments to the parser.
        """
        parser.add_argument("data", help="path to the directory of sign feature path. \
                            This directory should contains pre-computed features for sign videos.")
        parser.add_argument("-t", "--target-lang", default=None, metavar="TARGET", 
                            help="target language.")
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')

        # paths to pre-computed features
        # parser.add_argument('--src-body-feat-root', default=None, type=str,
        #                     help='path to to i3d feature for sign videos.')
        # parser.add_argument('--src-lefthand-feat-root', default=None, type=str,
        #                     help='path to left hand feature for sign videos.')
        # parser.add_argument('--src-righthand-feat-root', default=None, type=str,
        #                     help='path to right hand feature for sign videos.')
        parser.add_argument('--multilv-args', type=str, metavar='JSON',
                            help='multi-level feature aggregation args')
        parser.add_argument('--num-levels', default=-1, type=int,
                            help='total number of feature levels.')
        parser.add_argument('--src-lv0-body-feat-root', default=None, type=str,
                            help='path to to i3d feature for sign videos.')
        parser.add_argument('--src-lv0-lefthand-feat-root', default=None, type=str,
                            help='path to left hand feature for sign videos.')
        parser.add_argument('--src-lv0-righthand-feat-root', default=None, type=str,
                            help='path to right hand feature for sign videos.')
        parser.add_argument('--src-lv1-body-feat-root', default=None, type=str,
                            help='path to to i3d feature for sign videos.')
        parser.add_argument('--src-lv1-lefthand-feat-root', default=None, type=str,
                            help='path to left hand feature for sign videos.')
        parser.add_argument('--src-lv1-righthand-feat-root', default=None, type=str,
                            help='path to right hand feature for sign videos.')
        parser.add_argument('--src-lv2-body-feat-root', default=None, type=str,
                            help='path to to i3d feature for sign videos.')
        parser.add_argument('--src-lv2-lefthand-feat-root', default=None, type=str,
                            help='path to left hand feature for sign videos.')
        parser.add_argument('--src-lv2-righthand-feat-root', default=None, type=str,
                            help='path to right hand feature for sign videos.')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenizer before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='if setting, we compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        parser.add_argument('--eval-bleu-save', default=None, type=str,
                            help='path to save the hypothesis translations.')

        parser.add_argument('--disable-bucketing', action='store_true',
                            help='whether to disable bucketing in batching. If true, randomize all batches.')

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        if args.target_lang is None:
            raise Exception('Please provide target language explicitly.') 
       
        paths = args.data.split(os.pathsep)
        assert len(paths) > 0
        # parent = os.path.join(paths[:-1])

        if args.target_lang is None:
            raise Exception("Please provide target language explicitly.")
        target_lang = args.target_lang
        # dictionary file format expects '<token><space><count>', can be generated by fairseq-preprocess
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(target_lang)))

        logging.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, tgt_dict)
    
    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        paths = self.args.data.split(os.pathsep)

        multilv_args = json.loads(self.args.multilv_args)

        src, tgt = 'sign', self.args.target_lang

        assert len(paths) > 0
        data_path = paths[0]

        src_feat_roots = []
        max_lv = self.args.num_levels
        for lv in range(max_lv):
            prefix = 'src_lv' + str(lv)
            feature_root_names = ['{}_{}_feat_root'.format(prefix, feature_type)
                                  for feature_type in ['body', 'lefthand', 'righthand']]
            is_feature_root_level_specified = [
                (hasattr(self.args, feature_root_name) and getattr(self.args, feature_root_name) is not None)
                for feature_root_name in feature_root_names]
            if any(is_feature_root_level_specified):
                src_feat_roots.append([getattr(self.args, feature_root_name)
                                       for feature_root_name in feature_root_names])
            else:
                break

        self.datasets[split] = load_langpair_dataset(
            data_path, split,
            src, src_feat_roots,
            tgt, self.tgt_dict,
            dataset_impl=self.args.dataset_impl,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            use_bucketing=not self.args.disable_bucketing,
            multilv_args=multilv_args
        )

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def build_model(self, args):
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator(Namespace(**gen_args))
        return super().build_model(args)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu, hyps, refs = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len

            logging_output['hyps'] = hyps
            logging_output['refs'] = refs
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                escape_unk=escape_unk,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []

        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))

        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])

        tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args.eval_tokenized_bleu else 'none'
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize), hyps, refs
