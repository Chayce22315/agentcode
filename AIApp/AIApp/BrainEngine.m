#import "BrainEngine.h"
@import CoreML;

@implementation BrainEngine {
    MLModel *_Nullable _model;
    NSDictionary<NSString *, NSNumber *> *_Nullable _tokenToCol;
    NSArray<NSString *> *_Nullable _classes;
    NSArray<NSNumber *> *_Nullable _ngramRange;
    NSString *_Nullable _tokenPattern;
    NSString *_Nullable _inputName;
    BOOL _lowercase;
    NSInteger _featureCount;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        [self loadVocabularyIfNeeded];
        NSURL *url = [[NSBundle mainBundle] URLForResource:@"Brain" withExtension:@"mlmodelc"];
        if (url == nil) {
            url = [[NSBundle mainBundle] URLForResource:@"Brain" withExtension:@"mlpackage"];
        }
        if (url == nil) {
            return self;
        }
        NSError *error = nil;
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        _model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&error];
        if (_model == nil) {
            NSLog(@"BrainEngine: failed to load model: %@", error);
        }
    }
    return self;
}

- (void)loadVocabularyIfNeeded {
    if (_tokenToCol != nil) {
        return;
    }
    NSURL *url = [[NSBundle mainBundle] URLForResource:@"brain_vocab" withExtension:@"json"];
    if (url == nil) {
        NSLog(@"BrainEngine: brain_vocab.json missing from bundle");
        return;
    }
    NSData *data = [NSData dataWithContentsOfURL:url];
    if (data == nil) {
        return;
    }
    NSError *error = nil;
    id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![obj isKindOfClass:[NSDictionary class]] || error != nil) {
        NSLog(@"BrainEngine: invalid brain_vocab.json: %@", error);
        return;
    }
    NSDictionary *dict = (NSDictionary *)obj;
    NSArray *tokens = dict[@"tokens"];
    NSArray *classes = dict[@"classes"];
    NSArray *ngramRange = dict[@"ngram_range"];
    NSString *pattern = dict[@"token_pattern"];
    NSNumber *lowerNum = dict[@"lowercase"];
    NSString *inputName = dict[@"model_input_name"];
    NSNumber *fc = dict[@"feature_count"];

    if (![tokens isKindOfClass:[NSArray class]] || ![classes isKindOfClass:[NSArray class]]) {
        return;
    }
    _classes = classes;
    _ngramRange = [ngramRange isKindOfClass:[NSArray class]] ? ngramRange : @[ @1, @1 ];
    _tokenPattern = [pattern isKindOfClass:[NSString class]] ? pattern : @"(?u)\\b\\w\\w+\\b";
    _lowercase = lowerNum != nil ? lowerNum.boolValue : YES;
    _inputName = [inputName isKindOfClass:[NSString class]] ? inputName : @"features";
    _featureCount = fc != nil ? fc.integerValue : (NSInteger)tokens.count;

    NSMutableDictionary *map = [NSMutableDictionary dictionaryWithCapacity:tokens.count];
    NSInteger i = 0;
    for (id t in tokens) {
        if ([t isKindOfClass:[NSString class]] && [(NSString *)t length] > 0) {
            map[(NSString *)t] = @(i);
        }
        i++;
    }
    _tokenToCol = map;
}

/// Matches sklearn CountVectorizer default tokenization for training export (same regex + ngrams).
- (NSArray<NSString *> *)tokensFromText:(NSString *)text {
    NSString *work = _lowercase ? [text lowercaseString] : text;
    NSError *error = nil;
    NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:_tokenPattern
                                                                          options:0
                                                                            error:&error];
    if (re == nil) {
        NSLog(@"BrainEngine: bad token_pattern: %@", error);
        return @[];
    }
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    [re enumerateMatchesInString:work
                         options:0
                           range:NSMakeRange(0, work.length)
                      usingBlock:^(NSTextCheckingResult *result, NSMatchingFlags flags, BOOL *stop) {
                          [out addObject:[work substringWithRange:result.range]];
                      }];
    return out;
}

- (nullable MLMultiArray *)featureVectorForText:(NSString *)text {
    if (_tokenToCol == nil || _featureCount <= 0) {
        return nil;
    }
    NSArray<NSString *> *toks = [self tokensFromText:text];
    NSInteger nMin = _ngramRange != nil && _ngramRange.count >= 1 ? _ngramRange[0].integerValue : 1;
    NSInteger nMax = _ngramRange != nil && _ngramRange.count >= 2 ? _ngramRange[1].integerValue : 1;

    NSError *err = nil;
    MLMultiArray *vec = [[MLMultiArray alloc] initWithShape:@[ @(_featureCount) ]
                                                    dataType:MLMultiArrayDataTypeFloat32
                                                       error:&err];
    if (vec == nil) {
        NSLog(@"BrainEngine: MLMultiArray failed: %@", err);
        return nil;
    }

    void (^addToken)(NSString *) = ^(NSString *token) {
        NSNumber *col = self->_tokenToCol[token];
        if (col == nil) {
            return;
        }
        NSInteger idx = col.integerValue;
        if (idx < 0 || idx >= self->_featureCount) {
            return;
        }
        float *ptr = (float *)vec.dataPointer;
        ptr[idx] += 1.0f;
    };

    if (nMin <= 1) {
        for (NSString *t in toks) {
            addToken(t);
        }
    }
    if (nMax >= 2 && toks.count >= 2) {
        for (NSUInteger i = 0; i + 1 < toks.count; i++) {
            NSString *bigram = [NSString stringWithFormat:@"%@ %@", toks[i], toks[i + 1]];
            addToken(bigram);
        }
    }

    return vec;
}

- (nullable NSString *)predictIntent:(NSString *)text {
    if (_model == nil || text.length == 0) {
        return nil;
    }
    [self loadVocabularyIfNeeded];
    MLMultiArray *features = [self featureVectorForText:text];
    if (features == nil) {
        return nil;
    }

    NSString *inName = _inputName ?: @"features";
    MLFeatureValue *fv = [MLFeatureValue featureValueWithMultiArray:features];
    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
        inName : fv
    } error:nil];
    if (input == nil) {
        return nil;
    }

    NSError *error = nil;
    id<MLFeatureProvider> output = [_model predictionFromFeatures:input error:&error];
    if (output == nil) {
        NSLog(@"BrainEngine: prediction error: %@", error);
        return nil;
    }

    NSArray<NSString *> *names = @[ @"intent", @"classLabel", @"label" ];
    for (NSString *name in names) {
        MLFeatureValue *value = [output featureValueForName:name];
        if (value == nil) {
            continue;
        }
        if (value.type == MLFeatureTypeString) {
            return value.stringValue;
        }
    }
    return nil;
}

@end
