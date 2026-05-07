
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/__qc_index__.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}
require('./assets/3.16小游戏/command_TypeScript/exit');
require('./assets/3.16小游戏/command_TypeScript/level1/enemy');
require('./assets/3.16小游戏/command_TypeScript/level1/leftToRight');
require('./assets/3.16小游戏/command_TypeScript/level1/mian');
require('./assets/3.16小游戏/command_TypeScript/level2/enemy - 001');
require('./assets/3.16小游戏/command_TypeScript/level2/enemy - 002');
require('./assets/3.16小游戏/command_TypeScript/level2/enemy - 003');
require('./assets/3.16小游戏/command_TypeScript/level2/left');
require('./assets/3.16小游戏/command_TypeScript/level2/main');
require('./assets/3.16小游戏/command_TypeScript/level3/gloabl');
require('./assets/3.16小游戏/command_TypeScript/level3/main - 001');
require('./assets/3.16小游戏/command_TypeScript/level3/right');
require('./assets/3.16小游戏/command_TypeScript/level4/enemy1');
require('./assets/3.16小游戏/command_TypeScript/level4/enemy2');
require('./assets/3.16小游戏/command_TypeScript/level4/enemy3');
require('./assets/3.16小游戏/command_TypeScript/level4/global');
require('./assets/3.16小游戏/command_TypeScript/level4/main - 002');
require('./assets/3.16小游戏/command_TypeScript/level5/global - 001');
require('./assets/3.16小游戏/command_TypeScript/level5/left - 001');
require('./assets/3.16小游戏/command_TypeScript/level5/right - 001');
require('./assets/3.16小游戏/command_TypeScript/level6/global - 002');
require('./assets/3.16小游戏/command_TypeScript/level6/left - 002');
require('./assets/3.16小游戏/command_TypeScript/level6/right1');
require('./assets/3.16小游戏/command_TypeScript/level6/right2');
require('./assets/3.16小游戏/command_TypeScript/level6/right3');
require('./assets/3.16小游戏/command_TypeScript/level7/global - 003');
require('./assets/3.16小游戏/command_TypeScript/level7/left - 003');
require('./assets/3.16小游戏/command_TypeScript/level7/right1 - 001');
require('./assets/3.16小游戏/command_TypeScript/level7/right2 - 001');
require('./assets/3.16小游戏/command_TypeScript/level7/right3 - 001');
require('./assets/3.16小游戏/command_TypeScript/level8/global - 004');
require('./assets/3.16小游戏/command_TypeScript/level8/left - 004');
require('./assets/3.16小游戏/command_TypeScript/level8/right - 002');

                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();