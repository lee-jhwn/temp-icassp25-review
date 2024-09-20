
sents_all = []
filelists = ['../filelists/n400/filelist_train_text_ipa.txt', '../filelists/n400/filelist_test_unseen_audio_text_ipa.txt','../filelists/n400/filelist_test_unseen_subject_text_ipa.txt','../filelists/n400/filelist_test_unseen_both_text_ipa.txt']

for fn in filelists:
    with open(fn) as f:
        temp_fn = f.readlines()
        temp_fn = [_.split('||')[-1].rstrip() for _ in temp_fn]
    print(temp_fn)

    for line in temp_fn:
        sents_all.extend(list(line))

sents_all = ''.join(sorted(list(set(sents_all))))
print(sents_all)

