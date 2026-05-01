#import "BrainEngine.h"
@import CoreML;

@implementation BrainEngine {
    MLModel *_Nullable _model;
}

- (instancetype)init {
    self = [super init];
    if (self) {
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

- (nullable NSString *)predictIntent:(NSString *)text {
    if (_model == nil || text.length == 0) {
        return nil;
    }
    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
        @"text": text
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
