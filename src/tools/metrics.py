from whisper.normalizers import EnglishTextNormalizer
import jiwer


def eval_wer(hyps, refs, get_details=False):
    # assuming the texts are already aligned and there is no ID in the texts
    # WER
    std = EnglishTextNormalizer()
    hyps = [std(hyp) for hyp in hyps]
    refs = [std(ref) for ref in refs]
    out = jiwer.process_words(refs, hyps)
    
    total_ref = sum(len(ref.split()) for ref in refs)  # total number of words in the reference
    
    if not get_details:
        return out.wer
    else:
        # return ins, del and sub rates
        ins_rate = out.insertions / total_ref
        del_rate = out.deletions / total_ref
        sub_rate = out.substitutions / total_ref
        return {'WER': out.wer, 'INS': ins_rate, 'DEL': del_rate, 'SUB': sub_rate, 'HIT': out.hits/total_ref}

def eval_frac_0(hyps):
    '''
        Determine the fraction of samples that are 0 length
    '''
    return len([hyp for hyp in hyps if hyp==''])/len(hyps)