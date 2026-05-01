#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Loads the on-device Core ML model (exported from Python) and returns an intent label string.
@interface BrainEngine : NSObject

- (nullable NSString *)predictIntent:(NSString *)text;

@end

NS_ASSUME_NONNULL_END
