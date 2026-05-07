
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
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level1/enemy.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'df3cfPmM0ZEMpiVlGXpdno1', 'enemy');
// 3.16小游戏/command_TypeScript/level1/enemy.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var mian_1 = require("./mian");
var enemy = /** @class */ (function (_super) {
    __extends(enemy, _super);
    function enemy() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    enemy.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy.prototype.start = function () {
        this.schedule(function () {
            mian_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (mian_1.default.minion_attack == true) {
            mian_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        mian_1.default.minion_attack = false;
        if (mian_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(mian_1.default.main_hp, 19);
        }
    };
    enemy.prototype.update = function (dt) {
        if (mian_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    enemy = __decorate([
        ccclass
    ], enemy);
    return enemy;
}(cc.Component));
exports.default = enemy;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDFcXGVuZW15LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLCtCQUF5QjtBQUt6QjtJQUFtQyx5QkFBWTtJQUEvQzs7SUFxRUEsQ0FBQztJQWhFRyxzQkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBQzNCLENBQUM7SUFHRCxxQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLGNBQUksQ0FBQyxhQUFhLEdBQUcsSUFBSSxDQUFDO1FBQzlCLENBQUMsRUFBQyxDQUFDLENBQUMsQ0FBQztRQUVMLElBQUksQ0FBQyxRQUFRLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUM7SUFFRCxnQ0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFDdkIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ3pCLElBQUcsY0FBSSxDQUFDLGFBQWEsSUFBSSxJQUFJLEVBQUU7WUFDM0IsY0FBSSxDQUFDLE9BQU8sSUFBSSxDQUFDLENBQUM7WUFDbEIsSUFBSSxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyw2QkFBNkIsQ0FBQyxDQUFDO1lBQ3BELE1BQU0sQ0FBQyxZQUFZLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7WUFDNUMsSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQywrQkFBK0IsQ0FBQyxDQUFDO1lBQ3ZELE9BQU8sQ0FBQyxZQUFZLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUM7U0FDOUM7SUFJTCxDQUFDO0lBRUQsK0JBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2xCLEVBQUUsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7UUFHZCxjQUFJLENBQUMsYUFBYSxHQUFHLEtBQUssQ0FBQztRQUM1QixJQUFJLGNBQUksQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3RCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxjQUFJLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBRXpEO0lBR0osQ0FBQztJQUdELHNCQUFNLEdBQU4sVUFBUSxFQUFFO1FBT04sSUFBSSxjQUFJLENBQUMsYUFBYSxJQUFLLElBQUksRUFBRTtZQUM3QixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUksSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUUsQ0FBQztTQUdoRjthQUFNO1lBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsRUFBSTtnQkFDdkcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDM0U7U0FFSjtJQUNSLENBQUM7SUFwRWdCLEtBQUs7UUFEekIsT0FBTztPQUNhLEtBQUssQ0FxRXpCO0lBQUQsWUFBQztDQXJFRCxBQXFFQyxDQXJFa0MsRUFBRSxDQUFDLFNBQVMsR0FxRTlDO2tCQXJFb0IsS0FBSyIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIlxyXG4vLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuaW1wb3J0IG1haW4gZnJvbSBcIi4vbWlhblwiXHJcbmltcG9ydCBsZWZ0VG9SaWdodCBmcm9tIFwiLi9sZWZ0VG9SaWdodFwiXHJcblxyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgZW5lbXkgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgbWFpbi5taW5pb25fYXR0YWNrID0gdHJ1ZTtcclxuICAgICAgICB9LDEpO1xyXG5cclxuICAgICAgICB0aGlzLm1pbmlvbl94ID0gdGhpcy5ub2RlLnBvc2l0aW9uLng7XHJcbiAgICB9XHJcbiAgICBcclxuICAgIG9uQ29sbGlzaW9uRW50ZXIob3RoZXIsc2VsZil7XHJcbiAgICAgICAgY2MubG9nKFwi5byA5aeL56Kw5pKeXCIrb3RoZXIudGFnKTtcclxuICAgICAgICBpZihtYWluLm1pbmlvbl9hdHRhY2sgPT0gdHJ1ZSkge1xyXG4gICAgICAgICAgICBtYWluLm1haW5faHAgLT0gNTtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTVcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgIFxyXG4gICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgICAgY2MubG9nKFwi56Kw5pKe57uT5p2fXCIpO1xyXG4gICAgICAgIFxyXG4gICAgICBcclxuICAgICAgICBtYWluLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgIGlmKCBtYWluLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBtYWluLm1haW5faHAsIDE5KTtcclxuICAgICAgICBcclxuICAgICAgIH1cclxuICAgICAgXHJcbiAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG4gICAgICAgIFxyXG4gICAgICAgXHJcbiAgICAgICAgXHJcbiAgICAgICAgXHJcblxyXG5cclxuICAgICAgICBpZiAobWFpbi5taW5pb25fYXR0YWNrID09ICB0cnVlKSB7XHJcbiAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIFxyXG4gICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54ID49ICB0aGlzLm1pbmlvbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSAgdGhpcy5taW5pb25feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCArIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICBcclxuICAgICAgICAgICB9XHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/enemy - 002.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '8dbe6quYttEIJvOHgQwTjhz', 'enemy - 002');
// 3.16小游戏/command_TypeScript/level2/enemy - 002.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main_2 = require("./main");
var enemy_2 = /** @class */ (function (_super) {
    __extends(enemy_2, _super);
    function enemy_2() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_2.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_2.prototype.start = function () {
        this.schedule(function () {
            main_2.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_2.prototype.onCollisionEnter = function (other, self) {
        if (main_2.default.minion_attack == true) {
            main_2.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_2.prototype.onCollisionExit = function (other) {
        main_2.default.minion_attack = false;
        if (main_2.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(main_2.default.main_hp, 19);
        }
    };
    enemy_2.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼");
        if (this.count == 0) {
            if (node1.active == false) {
                var node2 = cc.find("Canvas/bj/b/a");
                node2.active = true;
                node2.setContentSize(200, 26);
                this.node.setPosition(this.node.position.x, node1.position.y);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (main_2.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_2 = __decorate([
        ccclass
    ], enemy_2);
    return enemy_2;
}(cc.Component));
exports.default = enemy_2;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXGVuZW15IC0gMDAyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLCtCQUEyQjtBQUszQjtJQUFxQywyQkFBWTtJQUFqRDtRQUFBLHFFQThFQztRQXpFSSxXQUFLLEdBQUcsQ0FBQyxDQUFDOztJQXlFZixDQUFDO0lBeEVHLHdCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFFM0IsQ0FBQztJQUdELHVCQUFLLEdBQUw7UUFDSSxJQUFJLENBQUMsUUFBUSxDQUFDO1lBQ1YsY0FBTSxDQUFDLGFBQWEsR0FBRyxJQUFJLENBQUM7UUFDaEMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBRUwsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQUVELGtDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUV2QixJQUFHLGNBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxDQUFDO1lBQ3BCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNwRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO1lBQzVDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN2RCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1NBQzlDO0lBR0wsQ0FBQztJQUVELGlDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUlsQixjQUFNLENBQUMsYUFBYSxHQUFHLEtBQUssQ0FBQztRQUM3QixJQUFJLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxjQUFNLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUdELHdCQUFNLEdBQU4sVUFBUSxFQUFFO1FBRU4sSUFBSSxLQUFLLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxtQkFBbUIsQ0FBQyxDQUFDO1FBQ3pDLElBQUksSUFBSSxDQUFDLEtBQUssSUFBSSxDQUFDLEVBQUU7WUFDakIsSUFBSSxLQUFLLENBQUMsTUFBTSxJQUFJLEtBQUssRUFBRTtnQkFDdkIsSUFBSSxLQUFLLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBRSxlQUFlLENBQUMsQ0FBQTtnQkFDckMsS0FBSyxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7Z0JBQ3BCLEtBQUssQ0FBQyxjQUFjLENBQUMsR0FBRyxFQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUM3QixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEVBQUcsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDakUsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO2FBQ2hCO1NBQ0o7UUFFRCxJQUFJLEtBQUssQ0FBQyxNQUFNLElBQUksS0FBSyxFQUFFO1lBQ3ZCLElBQUksY0FBTSxDQUFDLGFBQWEsSUFBSyxJQUFJLEVBQUU7Z0JBQy9CLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBSSxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBRSxDQUFDO2FBR2hGO2lCQUFNO2dCQUNILElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLEVBQUk7b0JBQ3ZHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2lCQUMzRTthQUVKO1NBQ1A7SUFHTCxDQUFDO0lBN0VnQixPQUFPO1FBRDNCLE9BQU87T0FDYSxPQUFPLENBOEUzQjtJQUFELGNBQUM7Q0E5RUQsQUE4RUMsQ0E5RW9DLEVBQUUsQ0FBQyxTQUFTLEdBOEVoRDtrQkE5RW9CLE9BQU8iLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyJcclxuLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcbmltcG9ydCBtYWluXzEgZnJvbSBcIi4vbWFpblwiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzIgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgICBjb3VudCA9IDA7XHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcblxyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMuc2NoZWR1bGUoKCkgPT4ge1xyXG4gICAgICAgICAgICBtYWluXzEubWluaW9uX2F0dGFjayA9IHRydWU7XHJcbiAgICAgICAgfSwxKTtcclxuXHJcbiAgICAgICAgdGhpcy5taW5pb25feCA9IHRoaXMubm9kZS5wb3NpdGlvbi54O1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYobWFpbl8xLm1pbmlvbl9hdHRhY2sgPT0gdHJ1ZSkge1xyXG4gICAgICAgICAgICBtYWluXzEubWFpbl9ocCAtPSA1O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItNVwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICBcclxuICAgICAgICBcclxuICAgICAgXHJcbiAgICAgICBtYWluXzEubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgICAgaWYoIG1haW5fMS5tYWluX2hwIDw9IDApIHtcclxuICAgICAgICBvdGhlci5ub2RlLmFjdGl2ZSA9IGZhbHNlO1xyXG4gICAgICAgIGxldCBsb3NlID0gY2MuZmluZChcIkNhbnZhcy9iai9mYWlsXCIpO1xyXG4gICAgICAgIGxvc2UuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZSggbWFpbl8xLm1haW5faHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuXHJcbiAgICB1cGRhdGUgKGR0KSB7XHJcbiAgICAgIFxyXG4gICAgICAgIGxldCBub2RlMSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi/lsI/prLxcIik7XHJcbiAgICAgICAgaWYgKHRoaXMuY291bnQgPT0gMCkge1xyXG4gICAgICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgICAgICBsZXQgbm9kZTIgPSBjYy5maW5kIChcIkNhbnZhcy9iai9iL2FcIilcclxuICAgICAgICAgICAgICAgIG5vZGUyLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICBub2RlMi5zZXRDb250ZW50U2l6ZSgyMDAsMjYpO1xyXG4gICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggLCBub2RlMS5wb3NpdGlvbi55KTsgICBcclxuICAgICAgICAgICAgICAgIHRoaXMuY291bnQrKzsgICAgICAgIFxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgIGlmIChub2RlMS5hY3RpdmUgPT0gZmFsc2UpIHtcclxuICAgICAgICAgICAgaWYgKG1haW5fMS5taW5pb25fYXR0YWNrID09ICB0cnVlKSB7XHJcbiAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24oICB0aGlzLm5vZGUucG9zaXRpb24ueCAgLSAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSApO1xyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gIHRoaXMubWluaW9uX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9ICB0aGlzLm1pbmlvbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCArIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBcclxuICAgIH1cclxufVxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level3/main - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'bb8a34UmdRErJ42JemF+/mT', 'main - 001');
// 3.16小游戏/command_TypeScript/level3/main - 001.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var gloabl_1 = require("./gloabl");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var mainCharacter = /** @class */ (function (_super) {
    __extends(mainCharacter, _super);
    function mainCharacter() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    mainCharacter.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    mainCharacter.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (gloabl_1.default.attack == true) {
            gloabl_1.default.minion1_hp -= 30;
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        gloabl_1.default.attack = false;
    };
    mainCharacter.prototype.onCollisionStay = function (other) {
        gloabl_1.default.attack = false;
        gloabl_1.default.minion_attack = false;
    };
    mainCharacter.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        if (gloabl_1.default.minion1_hp <= 0) {
            other.node.active = false;
            gloabl_1.default.attack = false;
            gloabl_1.default.minion_attack = false;
            gloabl_1.default.main_hp = 163;
            gloabl_1.default.minion1_hp = 163;
            gloabl_1.default.minion2_hp = 163;
            gloabl_1.default.minion3_hp = 163;
            cc.director.loadScene("fight4");
        }
        else {
            other.node.children[0].setContentSize(gloabl_1.default.minion1_hp, 19);
        }
    };
    mainCharacter.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    mainCharacter.prototype.update = function (dt) {
        if (gloabl_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], mainCharacter.prototype, "label", void 0);
    __decorate([
        property
    ], mainCharacter.prototype, "text", void 0);
    mainCharacter = __decorate([
        ccclass
    ], mainCharacter);
    return mainCharacter;
}(cc.Component));
exports.default = mainCharacter;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDNcXG1haW4gLSAwMDEudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsbUNBQTZCO0FBS3ZCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBbUZDO1FBaEZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUE2RTNCLENBQUM7SUF2RUcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFDdkIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRXpCLElBQUcsZ0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLGdCQUFNLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQztZQUN4QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUM3QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDckQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztRQUVELGdCQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztJQUMxQixDQUFDO0lBR0QsdUNBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2pCLGdCQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztRQUN0QixnQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7SUFDakMsQ0FBQztJQUVELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNsQixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBS2YsSUFBRyxnQkFBTSxDQUFDLFVBQVUsSUFBSSxDQUFDLEVBQUU7WUFDMUIsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLGdCQUFNLENBQUMsTUFBTSxHQUFFLEtBQUssQ0FBQztZQUNyQixnQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7WUFDN0IsZ0JBQU0sQ0FBQyxPQUFPLEdBQUUsR0FBRyxDQUFDO1lBQ3BCLGdCQUFNLENBQUMsVUFBVSxHQUFFLEdBQUcsQ0FBQztZQUN2QixnQkFBTSxDQUFDLFVBQVUsR0FBRSxHQUFHLENBQUM7WUFDdkIsZ0JBQU0sQ0FBQyxVQUFVLEdBQUcsR0FBRyxDQUFDO1lBQ3hCLEVBQUUsQ0FBQyxRQUFRLENBQUMsU0FBUyxDQUFDLFFBQVEsQ0FBQyxDQUFDO1NBQ2hDO2FBQU07WUFDTixLQUFLLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsZ0JBQU0sQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLENBQUM7U0FDN0Q7SUFFSixDQUFDO0lBRUQsNkJBQUssR0FBTDtRQUVBLElBQUksQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO0lBR25DLENBQUM7SUFFRCw4QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUNOLElBQUksZ0JBQU0sQ0FBQyxNQUFNLElBQUssSUFBSSxFQUFFO1lBQzNCLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBRzVFO2FBQU07WUFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQyxFQUFJO2dCQUNqRyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMzRTtTQUVKO0lBRUwsQ0FBQztJQTlFRDtRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO2dEQUNJO0lBR3ZCO1FBREMsUUFBUTsrQ0FDYztJQU5OLGFBQWE7UUFEakMsT0FBTztPQUNhLGFBQWEsQ0FtRmpDO0lBQUQsb0JBQUM7Q0FuRkQsQUFtRkMsQ0FuRjBDLEVBQUUsQ0FBQyxTQUFTLEdBbUZ0RDtrQkFuRm9CLGFBQWEiLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyIvLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcbmltcG9ydCBnbG9hYmwgZnJvbSBcIi4vZ2xvYWJsXCJcclxuXHJcblxyXG5cclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgbWFpbkNoYXJhY3RlciBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcbiAgICBtYWluX3g6IG51bWJlcjtcclxuICAgIFxyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3g6IG51bWJlcjtcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF95OiBudW1iZXI7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcblxyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgLy/kuqfnlJ/norDmkp7kvJrosIPnlKhcclxuICAgIG9uQ29sbGlzaW9uRW50ZXIob3RoZXIsc2VsZil7XHJcbiAgICAgICAgY2MubG9nKFwi5byA5aeL56Kw5pKeXCIrb3RoZXIudGFnKTtcclxuICAgICAgIFxyXG4gICAgICAgIGlmKGdsb2FibC5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIGdsb2FibC5taW5pb24xX2hwIC09IDMwO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL3h5L21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi0zMFwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBnbG9hYmwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG5cclxuICAgIG9uQ29sbGlzaW9uU3RheShvdGhlcikge1xyXG4gICAgICAgIGdsb2FibC5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBnbG9hYmwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgICAgY2MubG9nKFwi56Kw5pKe57uT5p2fXCIpO1xyXG4gICAgICAgXHJcbiAgICAgICBcclxuICAgICBcclxuICAgICAgXHJcbiAgICAgICBpZihnbG9hYmwubWluaW9uMV9ocCA8PSAwKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBnbG9hYmwuYXR0YWNrPSBmYWxzZTtcclxuICAgICAgICBnbG9hYmwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgICAgIGdsb2FibC5tYWluX2hwPSAxNjM7XHJcbiAgICAgICAgZ2xvYWJsLm1pbmlvbjFfaHA9IDE2MztcclxuICAgICAgICBnbG9hYmwubWluaW9uMl9ocD0gMTYzO1xyXG4gICAgICAgIGdsb2FibC5taW5pb24zX2hwID0gMTYzO1xyXG4gICAgICAgIGNjLmRpcmVjdG9yLmxvYWRTY2VuZShcImZpZ2h0NFwiKTtcclxuICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShnbG9hYmwubWluaW9uMV9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcbiAgICBcclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgXHJcbiAgICB0aGlzLm1haW5feCA9IHRoaXMubm9kZS5wb3NpdGlvbi54O1xyXG4gICAgICAgXHJcbiAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG4gICAgICAgIGlmIChnbG9hYmwuYXR0YWNrID09ICB0cnVlKSB7XHJcbiAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCArIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgXHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPD0gdGhpcy5tYWluX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54ID49IHRoaXMubWFpbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54IC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgIH1cclxuXHJcbiAgICB9XHJcbiAgICBcclxufVxyXG5cclxuXHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level4/enemy3.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '9502b8MZ7dH368gY8fXHg0+', 'enemy3');
// 3.16小游戏/command_TypeScript/level4/enemy3.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global_1 = require("./global");
var enemy_3 = /** @class */ (function (_super) {
    __extends(enemy_3, _super);
    function enemy_3() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_3.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_3.prototype.start = function () {
        this.schedule(function () {
            global_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_3.prototype.onCollisionEnter = function (other, self) {
        if (global_1.default.minion_attack == true) {
            global_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_3.prototype.onCollisionExit = function (other) {
        global_1.default.minion_attack = false;
        if (global_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global_1.default.main_hp, 19);
        }
    };
    enemy_3.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼2");
        if (this.count == 0) {
            if (node1.active == false) {
                this.node.setPosition(this.node.position.x, node1.position.y);
                var node2 = cc.find("Canvas/bj/b/a");
                node2.setContentSize(400, 26);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (global_1.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_3 = __decorate([
        ccclass
    ], enemy_3);
    return enemy_3;
}(cc.Component));
exports.default = enemy_3;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDRcXGVuZW15My50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQyxtQ0FBNkI7QUFLN0I7SUFBcUMsMkJBQVk7SUFBakQ7UUFBQSxxRUE4RUM7UUF6RUksV0FBSyxHQUFHLENBQUMsQ0FBQzs7SUF5RWYsQ0FBQztJQXhFRyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBRTNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLGdCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsa0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBRXZCLElBQUcsZ0JBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLGdCQUFNLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUM1QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztJQUdMLENBQUM7SUFFRCxpQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFJakIsZ0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzlCLElBQUksZ0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxnQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsb0JBQW9CLENBQUMsQ0FBQztRQUMxQyxJQUFJLElBQUksQ0FBQyxLQUFLLElBQUksQ0FBQyxFQUFFO1lBQ2pCLElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7Z0JBRXZCLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsRUFBRyxLQUFLLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNqRSxJQUFJLEtBQUssR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFFLGVBQWUsQ0FBQyxDQUFBO2dCQUNyQyxLQUFLLENBQUMsY0FBYyxDQUFDLEdBQUcsRUFBQyxFQUFFLENBQUMsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO2FBQ2hCO1NBQ0o7UUFFRCxJQUFJLEtBQUssQ0FBQyxNQUFNLElBQUksS0FBSyxFQUFFO1lBQ3ZCLElBQUksZ0JBQU0sQ0FBQyxhQUFhLElBQUssSUFBSSxFQUFFO2dCQUMvQixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUksSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUUsQ0FBQzthQUdoRjtpQkFBTTtnQkFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxFQUFJO29CQUN2RyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztpQkFDM0U7YUFFSjtTQUNQO0lBR0wsQ0FBQztJQTdFZ0IsT0FBTztRQUQzQixPQUFPO09BQ2EsT0FBTyxDQThFM0I7SUFBRCxjQUFDO0NBOUVELEFBOEVDLENBOUVvQyxFQUFFLENBQUMsU0FBUyxHQThFaEQ7a0JBOUVvQixPQUFPIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbFwiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzMgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgICBjb3VudCA9IDA7XHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcblxyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMuc2NoZWR1bGUoKCkgPT4ge1xyXG4gICAgICAgICAgICBnbG9iYWwubWluaW9uX2F0dGFjayA9IHRydWU7XHJcbiAgICAgICAgfSwxKTtcclxuXHJcbiAgICAgICAgdGhpcy5taW5pb25feCA9IHRoaXMubm9kZS5wb3NpdGlvbi54O1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gdHJ1ZSkge1xyXG4gICAgICAgICAgICBnbG9iYWwubWFpbl9ocCAtPSA1O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItNVwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgXHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgIGdsb2JhbC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYmFsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9iYWwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgXHJcbiAgICAgICAgbGV0IG5vZGUxID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL+Wwj+msvDJcIik7XHJcbiAgICAgICAgaWYgKHRoaXMuY291bnQgPT0gMCkge1xyXG4gICAgICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICwgbm9kZTEucG9zaXRpb24ueSk7ICAgXHJcbiAgICAgICAgICAgICAgICBsZXQgbm9kZTIgPSBjYy5maW5kIChcIkNhbnZhcy9iai9iL2FcIilcclxuICAgICAgICAgICAgICAgIG5vZGUyLnNldENvbnRlbnRTaXplKDQwMCwyNik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLmNvdW50Kys7ICAgICAgICBcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgIGlmIChnbG9iYWwubWluaW9uX2F0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54ID49ICB0aGlzLm1pbmlvbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSAgdGhpcy5taW5pb25feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgXHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level5/left - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'ab722XkV0BFB5hi6AcvAkW1', 'left - 001');
// 3.16小游戏/command_TypeScript/level5/left - 001.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global___001_1 = require("./global - 001");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var mainCharacter = /** @class */ (function (_super) {
    __extends(mainCharacter, _super);
    function mainCharacter() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    mainCharacter.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    mainCharacter.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___001_1.default.attack == true) {
            global___001_1.default.minion1_hp -= 17;
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        global___001_1.default.attack = false;
    };
    mainCharacter.prototype.onCollisionStay = function (other) {
        global___001_1.default.attack = false;
        global___001_1.default.minion_attack = false;
    };
    mainCharacter.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        if (global___001_1.default.minion1_hp <= 0) {
            other.node.active = false;
            global___001_1.default.attack = false;
            global___001_1.default.minion_attack = false;
            global___001_1.default.main_hp = 163;
            global___001_1.default.minion1_hp = 163;
            global___001_1.default.minion2_hp = 163;
            global___001_1.default.minion3_hp = 163;
            cc.director.loadScene("fight6");
        }
        else {
            other.node.children[0].setContentSize(global___001_1.default.minion1_hp, 19);
        }
    };
    mainCharacter.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    mainCharacter.prototype.update = function (dt) {
        if (global___001_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], mainCharacter.prototype, "label", void 0);
    __decorate([
        property
    ], mainCharacter.prototype, "text", void 0);
    mainCharacter = __decorate([
        ccclass
    ], mainCharacter);
    return mainCharacter;
}(cc.Component));
exports.default = mainCharacter;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDVcXGxlZnQgLSAwMDEudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsK0NBQW1DO0FBSzdCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBK0VDO1FBNUVHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUF5RTNCLENBQUM7SUFuRUcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFDdkIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRXpCLElBQUcsc0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLHNCQUFNLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQztZQUN4QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUM3QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDckQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztRQUVELHNCQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztJQUMxQixDQUFDO0lBR0QsdUNBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2pCLHNCQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztRQUN0QixzQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7SUFDakMsQ0FBQztJQUVELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNsQixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ2YsSUFBRyxzQkFBTSxDQUFDLFVBQVUsSUFBSSxDQUFDLEVBQUU7WUFDMUIsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLHNCQUFNLENBQUMsTUFBTSxHQUFFLEtBQUssQ0FBQztZQUNyQixzQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7WUFDN0Isc0JBQU0sQ0FBQyxPQUFPLEdBQUUsR0FBRyxDQUFDO1lBQ3BCLHNCQUFNLENBQUMsVUFBVSxHQUFFLEdBQUcsQ0FBQztZQUN2QixzQkFBTSxDQUFDLFVBQVUsR0FBRSxHQUFHLENBQUM7WUFDdkIsc0JBQU0sQ0FBQyxVQUFVLEdBQUcsR0FBRyxDQUFDO1lBQ3hCLEVBQUUsQ0FBQyxRQUFRLENBQUMsU0FBUyxDQUFDLFFBQVEsQ0FBQyxDQUFDO1NBQ2hDO2FBQU07WUFDTixLQUFLLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsc0JBQU0sQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLENBQUM7U0FDN0Q7SUFFSixDQUFDO0lBRUQsNkJBQUssR0FBTDtRQUVBLElBQUksQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO0lBR25DLENBQUM7SUFFRCw4QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUNOLElBQUksc0JBQU0sQ0FBQyxNQUFNLElBQUssSUFBSSxFQUFFO1lBQzNCLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBRzVFO2FBQU07WUFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQyxFQUFJO2dCQUNqRyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMzRTtTQUVKO0lBRUwsQ0FBQztJQTFFRDtRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO2dEQUNJO0lBR3ZCO1FBREMsUUFBUTsrQ0FDYztJQU5OLGFBQWE7UUFEakMsT0FBTztPQUNhLGFBQWEsQ0ErRWpDO0lBQUQsb0JBQUM7Q0EvRUQsQUErRUMsQ0EvRTBDLEVBQUUsQ0FBQyxTQUFTLEdBK0V0RDtrQkEvRW9CLGFBQWEiLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyIvLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcbmltcG9ydCBnbG9hYmwgZnJvbSBcIi4vZ2xvYmFsIC0gMDAxXCJcclxuXHJcblxyXG5cclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgbWFpbkNoYXJhY3RlciBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcbiAgICBtYWluX3g6IG51bWJlcjtcclxuICAgIFxyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3g6IG51bWJlcjtcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF95OiBudW1iZXI7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcblxyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgLy/kuqfnlJ/norDmkp7kvJrosIPnlKhcclxuICAgIG9uQ29sbGlzaW9uRW50ZXIob3RoZXIsc2VsZil7XHJcbiAgICAgICAgY2MubG9nKFwi5byA5aeL56Kw5pKeXCIrb3RoZXIudGFnKTtcclxuICAgICAgIFxyXG4gICAgICAgIGlmKGdsb2FibC5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIGdsb2FibC5taW5pb24xX2hwIC09IDE3O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL3h5L21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi0zMFwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBnbG9hYmwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG5cclxuICAgIG9uQ29sbGlzaW9uU3RheShvdGhlcikge1xyXG4gICAgICAgIGdsb2FibC5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBnbG9hYmwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgICAgY2MubG9nKFwi56Kw5pKe57uT5p2fXCIpO1xyXG4gICAgICAgaWYoZ2xvYWJsLm1pbmlvbjFfaHAgPD0gMCkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgZ2xvYWJsLmF0dGFjaz0gZmFsc2U7XHJcbiAgICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBnbG9hYmwubWFpbl9ocD0gMTYzO1xyXG4gICAgICAgIGdsb2FibC5taW5pb24xX2hwPSAxNjM7XHJcbiAgICAgICAgZ2xvYWJsLm1pbmlvbjJfaHA9IDE2MztcclxuICAgICAgICBnbG9hYmwubWluaW9uM19ocCA9IDE2MztcclxuICAgICAgICBjYy5kaXJlY3Rvci5sb2FkU2NlbmUoXCJmaWdodDZcIik7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoZ2xvYWJsLm1pbmlvbjFfaHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgIFxyXG4gICAgdGhpcy5tYWluX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgIFxyXG4gICAgIFxyXG4gICAgfVxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBpZiAoZ2xvYWJsLmF0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgIFxyXG4gICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9IHRoaXMubWFpbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSB0aGlzLm1haW5feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBcclxuICAgICAgICB9XHJcblxyXG4gICAgfVxyXG4gICAgXHJcbn1cclxuXHJcblxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level6/left - 002.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'c2223j/Y0NO0oEAH2vDLDf5', 'left - 002');
// 3.16小游戏/command_TypeScript/level6/left - 002.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global___002_1 = require("./global - 002");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight_1 = /** @class */ (function (_super) {
    __extends(leftToRight_1, _super);
    function leftToRight_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    leftToRight_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight_1.prototype.onCollisionEnter = function (other, self) {
        cc.log(other.node.name);
        if (global___002_1.default.attack == true) {
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            if (other.node.name == "小鬼") {
                global___002_1.default.minion1_hp -= 30;
            }
            else if (other.node.name == "小鬼2") {
                global___002_1.default.minion2_hp -= 30;
            }
            else if (other.node.name == "小鬼3") {
                global___002_1.default.minion3_hp -= 30;
            }
        }
        global___002_1.default.attack = false;
    };
    leftToRight_1.prototype.onCollisionStay = function (other) {
        global___002_1.default.attack = false;
        global___002_1.default.minion_attack = false;
    };
    leftToRight_1.prototype.onCollisionExit = function (other) {
        if ((global___002_1.default.minion1_hp <= 0) && (other.node.name == "小鬼")) {
            other.node.active = false;
        }
        else if ((global___002_1.default.minion2_hp <= 0) && (other.node.name == "小鬼2")) {
            other.node.active = false;
        }
        else if ((global___002_1.default.minion3_hp <= 0) && (other.node.name == "小鬼3")) {
            other.node.active = false;
            cc.director.loadScene("fight7");
        }
        else {
            if (other.node.name == "小鬼") {
                other.node.children[0].setContentSize(global___002_1.default.minion1_hp, 19);
            }
            else if (other.node.name == "小鬼2") {
                other.node.children[0].setContentSize(global___002_1.default.minion2_hp, 19);
            }
            else if (other.node.name == "小鬼3") {
                other.node.children[0].setContentSize(global___002_1.default.minion3_hp, 19);
            }
        }
    };
    leftToRight_1.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight_1.prototype.update = function (dt) {
        if (global___002_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], leftToRight_1.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight_1.prototype, "text", void 0);
    leftToRight_1 = __decorate([
        ccclass
    ], leftToRight_1);
    return leftToRight_1;
}(cc.Component));
exports.default = leftToRight_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDZcXGxlZnQgLSAwMDIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsK0NBQW1DO0FBSzdCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBOEZDO1FBM0ZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUF3RjNCLENBQUM7SUFsRkcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFFeEIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3ZCLElBQUcsc0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN0RCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzdDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNyRCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1lBQzVDLElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxFQUFFO2dCQUMzQixzQkFBTSxDQUFDLFVBQVUsSUFBSSxFQUFFLENBQUM7YUFDeEI7aUJBQU8sSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7Z0JBQ3BDLHNCQUFNLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQzthQUN4QjtpQkFBTyxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssRUFBRTtnQkFDcEMsc0JBQU0sQ0FBQyxVQUFVLElBQUksRUFBRSxDQUFDO2FBQ3hCO1NBQ0g7UUFFRCxzQkFBTSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7SUFDMUIsQ0FBQztJQUdELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNqQixzQkFBTSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7UUFDdEIsc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO0lBQ2pDLENBQUM7SUFFRCx1Q0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFNbEIsSUFBRyxDQUFDLHNCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLEVBQUU7WUFDdkQsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQzFCO2FBQU8sSUFBRyxDQUFDLHNCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDaEUsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQzFCO2FBQU0sSUFBRyxDQUFDLHNCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDL0QsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLEVBQUUsQ0FBQyxRQUFRLENBQUMsU0FBUyxDQUFDLFFBQVEsQ0FBQyxDQUFDO1NBQ2hDO2FBQU07WUFDSCxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLElBQUksRUFBRTtnQkFDM0IsS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFDLHNCQUFNLENBQUMsVUFBVSxFQUFFLEVBQUUsQ0FBQyxDQUFDO2FBQzdEO2lCQUFPLElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxFQUFFO2dCQUNwQyxLQUFLLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsc0JBQU0sQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLENBQUM7YUFDN0Q7aUJBQU8sSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7Z0JBQ3BDLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxzQkFBTSxDQUFDLFVBQVUsRUFBRSxFQUFFLENBQUMsQ0FBQzthQUM3RDtTQUNKO0lBRUosQ0FBQztJQUVELDZCQUFLLEdBQUw7UUFFQSxJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUduQyxDQUFDO0lBRUQsOEJBQU0sR0FBTixVQUFRLEVBQUU7UUFDTixJQUFJLHNCQUFNLENBQUMsTUFBTSxJQUFLLElBQUksRUFBRTtZQUMzQixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUc1RTthQUFNO1lBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUMsRUFBSTtnQkFDakcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDM0U7U0FFSjtJQUdMLENBQUM7SUF6RkQ7UUFEQyxRQUFRLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQztnREFDSTtJQUd2QjtRQURDLFFBQVE7K0NBQ2M7SUFOTixhQUFhO1FBRGpDLE9BQU87T0FDYSxhQUFhLENBOEZqQztJQUFELG9CQUFDO0NBOUZELEFBOEZDLENBOUYwQyxFQUFFLENBQUMsU0FBUyxHQThGdEQ7a0JBOUZvQixhQUFhIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbCAtIDAwMlwiXHJcblxyXG5cclxuXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGxlZnRUb1JpZ2h0XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG5cclxuICAgIEBwcm9wZXJ0eShjYy5MYWJlbClcclxuICAgIGxhYmVsOiBjYy5MYWJlbCA9IG51bGw7XHJcblxyXG4gICAgQHByb3BlcnR5XHJcbiAgICB0ZXh0OiBzdHJpbmcgPSAnaGVsbG8nO1xyXG4gICAgbWFpbl94OiBudW1iZXI7XHJcbiAgICBcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF94OiBudW1iZXI7XHJcbiAgICBwdWJsaWMgc3RhdGljIGN1cnJlbnRfeTogbnVtYmVyO1xyXG5cclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG5cclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIC8v5Lqn55Sf56Kw5pKe5Lya6LCD55SoXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICBcclxuICAgICAgIGNjLmxvZyhvdGhlci5ub2RlLm5hbWUpO1xyXG4gICAgICAgIGlmKGdsb2JhbC5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTMwXCI7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UyID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgICAgaWYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8XCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjFfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjJfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwzXCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjNfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBnbG9iYWwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG5cclxuICAgIG9uQ29sbGlzaW9uU3RheShvdGhlcikge1xyXG4gICAgICAgIGdsb2JhbC5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBnbG9iYWwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG5cclxuICAgICAgIFxyXG4gICAgICAgXHJcbiAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgaWYoKGdsb2JhbC5taW5pb24xX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8XCIpKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgIH0gIGVsc2UgaWYoKGdsb2JhbC5taW5pb24yX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8MlwiKSkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICB9IGVsc2UgaWYoKGdsb2JhbC5taW5pb24zX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8M1wiKSkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgY2MuZGlyZWN0b3IubG9hZFNjZW5lKFwiZmlnaHQ3XCIpO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLxcIikge1xyXG4gICAgICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKGdsb2JhbC5taW5pb24xX2hwLCAxOSk7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpIHtcclxuICAgICAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShnbG9iYWwubWluaW9uMl9ocCwgMTkpO1xyXG4gICAgICAgICAgIH0gZWxzZSAgaWYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8M1wiKSB7XHJcbiAgICAgICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoZ2xvYmFsLm1pbmlvbjNfaHAsIDE5KTtcclxuICAgICAgICAgICB9XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuICAgIFxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICBcclxuICAgIHRoaXMubWFpbl94ID0gdGhpcy5ub2RlLnBvc2l0aW9uLng7XHJcbiAgICAgICBcclxuICAgICBcclxuICAgIH1cclxuXHJcbiAgICB1cGRhdGUgKGR0KSB7XHJcbiAgICAgICAgaWYgKGdsb2JhbC5hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICBcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSB0aGlzLm1haW5feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gdGhpcy5tYWluX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggLSAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgfVxyXG5cclxuICAgICBcclxuICAgIH1cclxuICAgIFxyXG59XHJcblxyXG5cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level7/left - 003.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '94ac6+/JHVNbJButEAfFRz4', 'left - 003');
// 3.16小游戏/command_TypeScript/level7/left - 003.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global___003_1 = require("./global - 003");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight_1 = /** @class */ (function (_super) {
    __extends(leftToRight_1, _super);
    function leftToRight_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    leftToRight_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight_1.prototype.onCollisionEnter = function (other, self) {
        cc.log(other.node.name);
        if (global___003_1.default.attack == true) {
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            if (other.node.name == "小鬼") {
                global___003_1.default.minion1_hp -= 30;
            }
            else if (other.node.name == "小鬼2") {
                global___003_1.default.minion2_hp -= 30;
            }
            else if (other.node.name == "小鬼3") {
                global___003_1.default.minion3_hp -= 30;
            }
        }
        global___003_1.default.attack = false;
    };
    leftToRight_1.prototype.onCollisionStay = function (other) {
        global___003_1.default.attack = false;
        global___003_1.default.minion_attack = false;
    };
    leftToRight_1.prototype.onCollisionExit = function (other) {
        if ((global___003_1.default.minion1_hp <= 0) && (other.node.name == "小鬼")) {
            other.node.active = false;
        }
        else if ((global___003_1.default.minion2_hp <= 0) && (other.node.name == "小鬼2")) {
            other.node.active = false;
        }
        else if ((global___003_1.default.minion3_hp <= 0) && (other.node.name == "小鬼3")) {
            other.node.active = false;
            cc.director.loadScene("fight8");
        }
        else {
            if (other.node.name == "小鬼") {
                other.node.children[0].setContentSize(global___003_1.default.minion1_hp, 19);
            }
            else if (other.node.name == "小鬼2") {
                other.node.children[0].setContentSize(global___003_1.default.minion2_hp, 19);
            }
            else if (other.node.name == "小鬼3") {
                other.node.children[0].setContentSize(global___003_1.default.minion3_hp, 19);
            }
        }
    };
    leftToRight_1.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight_1.prototype.update = function (dt) {
        if (global___003_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], leftToRight_1.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight_1.prototype, "text", void 0);
    leftToRight_1 = __decorate([
        ccclass
    ], leftToRight_1);
    return leftToRight_1;
}(cc.Component));
exports.default = leftToRight_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDdcXGxlZnQgLSAwMDMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsK0NBQW1DO0FBSzdCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBOEZDO1FBM0ZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUF3RjNCLENBQUM7SUFsRkcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFFeEIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3ZCLElBQUcsc0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN0RCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzdDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNyRCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1lBQzVDLElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxFQUFFO2dCQUMzQixzQkFBTSxDQUFDLFVBQVUsSUFBSSxFQUFFLENBQUM7YUFDeEI7aUJBQU8sSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7Z0JBQ3BDLHNCQUFNLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQzthQUN4QjtpQkFBTyxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssRUFBRTtnQkFDcEMsc0JBQU0sQ0FBQyxVQUFVLElBQUksRUFBRSxDQUFDO2FBQ3hCO1NBQ0g7UUFFRCxzQkFBTSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7SUFDMUIsQ0FBQztJQUdELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNqQixzQkFBTSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7UUFDdEIsc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO0lBQ2pDLENBQUM7SUFFRCx1Q0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFNbEIsSUFBRyxDQUFDLHNCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLEVBQUU7WUFDdkQsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQzFCO2FBQU8sSUFBRyxDQUFDLHNCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDaEUsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQzFCO2FBQU0sSUFBRyxDQUFDLHNCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDL0QsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLEVBQUUsQ0FBQyxRQUFRLENBQUMsU0FBUyxDQUFDLFFBQVEsQ0FBQyxDQUFDO1NBQ2hDO2FBQU07WUFDSCxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLElBQUksRUFBRTtnQkFDM0IsS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFDLHNCQUFNLENBQUMsVUFBVSxFQUFFLEVBQUUsQ0FBQyxDQUFDO2FBQzdEO2lCQUFPLElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxFQUFFO2dCQUNwQyxLQUFLLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsc0JBQU0sQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLENBQUM7YUFDN0Q7aUJBQU8sSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7Z0JBQ3BDLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxzQkFBTSxDQUFDLFVBQVUsRUFBRSxFQUFFLENBQUMsQ0FBQzthQUM3RDtTQUNKO0lBRUosQ0FBQztJQUVELDZCQUFLLEdBQUw7UUFFQSxJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUduQyxDQUFDO0lBRUQsOEJBQU0sR0FBTixVQUFRLEVBQUU7UUFDTixJQUFJLHNCQUFNLENBQUMsTUFBTSxJQUFLLElBQUksRUFBRTtZQUMzQixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUc1RTthQUFNO1lBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUMsRUFBSTtnQkFDakcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDM0U7U0FFSjtJQUdMLENBQUM7SUF6RkQ7UUFEQyxRQUFRLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQztnREFDSTtJQUd2QjtRQURDLFFBQVE7K0NBQ2M7SUFOTixhQUFhO1FBRGpDLE9BQU87T0FDYSxhQUFhLENBOEZqQztJQUFELG9CQUFDO0NBOUZELEFBOEZDLENBOUYwQyxFQUFFLENBQUMsU0FBUyxHQThGdEQ7a0JBOUZvQixhQUFhIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbCAtIDAwM1wiXHJcblxyXG5cclxuXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGxlZnRUb1JpZ2h0XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG5cclxuICAgIEBwcm9wZXJ0eShjYy5MYWJlbClcclxuICAgIGxhYmVsOiBjYy5MYWJlbCA9IG51bGw7XHJcblxyXG4gICAgQHByb3BlcnR5XHJcbiAgICB0ZXh0OiBzdHJpbmcgPSAnaGVsbG8nO1xyXG4gICAgbWFpbl94OiBudW1iZXI7XHJcbiAgICBcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF94OiBudW1iZXI7XHJcbiAgICBwdWJsaWMgc3RhdGljIGN1cnJlbnRfeTogbnVtYmVyO1xyXG5cclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG5cclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIC8v5Lqn55Sf56Kw5pKe5Lya6LCD55SoXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICBcclxuICAgICAgIGNjLmxvZyhvdGhlci5ub2RlLm5hbWUpO1xyXG4gICAgICAgIGlmKGdsb2JhbC5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTMwXCI7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UyID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgICAgaWYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8XCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjFfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjJfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwzXCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjNfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBnbG9iYWwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG5cclxuICAgIG9uQ29sbGlzaW9uU3RheShvdGhlcikge1xyXG4gICAgICAgIGdsb2JhbC5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBnbG9iYWwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG5cclxuICAgICAgIFxyXG4gICAgICAgXHJcbiAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgaWYoKGdsb2JhbC5taW5pb24xX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8XCIpKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgIH0gIGVsc2UgaWYoKGdsb2JhbC5taW5pb24yX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8MlwiKSkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICB9IGVsc2UgaWYoKGdsb2JhbC5taW5pb24zX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8M1wiKSkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgY2MuZGlyZWN0b3IubG9hZFNjZW5lKFwiZmlnaHQ4XCIpO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLxcIikge1xyXG4gICAgICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKGdsb2JhbC5taW5pb24xX2hwLCAxOSk7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpIHtcclxuICAgICAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShnbG9iYWwubWluaW9uMl9ocCwgMTkpO1xyXG4gICAgICAgICAgIH0gZWxzZSAgaWYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8M1wiKSB7XHJcbiAgICAgICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoZ2xvYmFsLm1pbmlvbjNfaHAsIDE5KTtcclxuICAgICAgICAgICB9XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuICAgIFxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICBcclxuICAgIHRoaXMubWFpbl94ID0gdGhpcy5ub2RlLnBvc2l0aW9uLng7XHJcbiAgICAgICBcclxuICAgICBcclxuICAgIH1cclxuXHJcbiAgICB1cGRhdGUgKGR0KSB7XHJcbiAgICAgICAgaWYgKGdsb2JhbC5hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICBcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSB0aGlzLm1haW5feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gdGhpcy5tYWluX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggLSAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgfVxyXG5cclxuICAgICBcclxuICAgIH1cclxuICAgIFxyXG59XHJcblxyXG5cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level8/left - 004.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '89eebWIFaNKq4HsEwTz0YAZ', 'left - 004');
// 3.16小游戏/command_TypeScript/level8/left - 004.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global___004_1 = require("./global - 004");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var mainCharacter = /** @class */ (function (_super) {
    __extends(mainCharacter, _super);
    function mainCharacter() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    mainCharacter.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    mainCharacter.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___004_1.default.attack == true) {
            global___004_1.default.minion1_hp -= 999;
            var damage = cc.find("Canvas/bj/kuan/main_damage");
            damage.getComponent(cc.Label).string = "-999";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        global___004_1.default.attack = false;
    };
    mainCharacter.prototype.onCollisionStay = function (other) {
        global___004_1.default.attack = false;
        global___004_1.default.minion_attack = false;
    };
    mainCharacter.prototype.onCollisionExit = function (other, self) {
        cc.log("碰撞结束");
        if (global___004_1.default.minion1_hp <= 0) {
            other.node.active = false;
            global___004_1.default.attack = false;
            global___004_1.default.minion_attack = false;
            var damage = cc.find("Canvas/bj/kuan/main_damage");
            damage.getComponent(cc.Label).string = "";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            var node1 = cc.find("Canvas/bj/d-8/c-8");
            node1.active = true;
            var node2 = cc.find("Canvas/bj/b/a");
            node2.active = true;
            self.node.active = false;
            var node3 = cc.find("Canvas/bj/kuan/tgcg");
            node3.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___004_1.default.minion1_hp, 19);
        }
    };
    mainCharacter.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    mainCharacter.prototype.update = function (dt) {
        if (global___004_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], mainCharacter.prototype, "label", void 0);
    __decorate([
        property
    ], mainCharacter.prototype, "text", void 0);
    mainCharacter = __decorate([
        ccclass
    ], mainCharacter);
    return mainCharacter;
}(cc.Component));
exports.default = mainCharacter;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDhcXGxlZnQgLSAwMDQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsK0NBQW1DO0FBSzdCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBcUZDO1FBbEZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUErRTNCLENBQUM7SUF6RUcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFDdkIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRXpCLElBQUcsc0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLHNCQUFNLENBQUMsVUFBVSxJQUFJLEdBQUcsQ0FBQztZQUN6QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDRCQUE0QixDQUFDLENBQUM7WUFDbkQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQztZQUM5QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDckQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztRQUVELHNCQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztJQUMxQixDQUFDO0lBR0QsdUNBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2pCLHNCQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztRQUN0QixzQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7SUFDakMsQ0FBQztJQUVELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSyxFQUFDLElBQUk7UUFDdkIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUNmLElBQUcsc0JBQU0sQ0FBQyxVQUFVLElBQUksQ0FBQyxFQUFFO1lBQzFCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixzQkFBTSxDQUFDLE1BQU0sR0FBRSxLQUFLLENBQUM7WUFDckIsc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1lBQzdCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNEJBQTRCLENBQUMsQ0FBQztZQUNuRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1lBQzFDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNyRCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1lBQzNDLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsbUJBQW1CLENBQUMsQ0FBQztZQUN6QyxLQUFLLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUNwQixJQUFJLEtBQUssR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxDQUFDO1lBQ3JDLEtBQUssQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO1lBQ3BCLElBQUksQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUN6QixJQUFJLEtBQUssR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLHFCQUFxQixDQUFDLENBQUM7WUFDM0MsS0FBSyxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDcEI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxzQkFBTSxDQUFDLFVBQVUsRUFBRSxFQUFFLENBQUMsQ0FBQztTQUM3RDtJQUVKLENBQUM7SUFFRCw2QkFBSyxHQUFMO1FBRUEsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFHbkMsQ0FBQztJQUVELDhCQUFNLEdBQU4sVUFBUSxFQUFFO1FBQ04sSUFBSSxzQkFBTSxDQUFDLE1BQU0sSUFBSyxJQUFJLEVBQUU7WUFDM0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FHNUU7YUFBTTtZQUNILElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDLEVBQUk7Z0JBQ2pHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzNFO1NBRUo7SUFFTCxDQUFDO0lBaEZEO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7Z0RBQ0k7SUFHdkI7UUFEQyxRQUFROytDQUNjO0lBTk4sYUFBYTtRQURqQyxPQUFPO09BQ2EsYUFBYSxDQXFGakM7SUFBRCxvQkFBQztDQXJGRCxBQXFGQyxDQXJGMEMsRUFBRSxDQUFDLFNBQVMsR0FxRnREO2tCQXJGb0IsYUFBYSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuaW1wb3J0IGdsb2FibCBmcm9tIFwiLi9nbG9iYWwgLSAwMDRcIlxyXG5cclxuXHJcblxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBtYWluQ2hhcmFjdGVyIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuXHJcbiAgICBAcHJvcGVydHkoY2MuTGFiZWwpXHJcbiAgICBsYWJlbDogY2MuTGFiZWwgPSBudWxsO1xyXG5cclxuICAgIEBwcm9wZXJ0eVxyXG4gICAgdGV4dDogc3RyaW5nID0gJ2hlbGxvJztcclxuICAgIG1haW5feDogbnVtYmVyO1xyXG4gICAgXHJcbiAgICBwdWJsaWMgc3RhdGljIGN1cnJlbnRfeDogbnVtYmVyO1xyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3k6IG51bWJlcjtcclxuXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuXHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcbiAgICB9XHJcbiAgICBcclxuXHJcbiAgICAvL+S6p+eUn+eisOaSnuS8muiwg+eUqFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYoZ2xvYWJsLmF0dGFjayA9PSB0cnVlKXtcclxuICAgICAgICAgICAgZ2xvYWJsLm1pbmlvbjFfaHAgLT0gOTk5O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi05OTlcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgZ2xvYWJsLmF0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuXHJcbiAgICBvbkNvbGxpc2lvblN0YXkob3RoZXIpIHtcclxuICAgICAgICBnbG9hYmwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgIH1cclxuXHJcbiAgICBvbkNvbGxpc2lvbkV4aXQob3RoZXIsc2VsZikge1xyXG4gICAgICAgY2MubG9nKFwi56Kw5pKe57uT5p2fXCIpO1xyXG4gICAgICAgaWYoZ2xvYWJsLm1pbmlvbjFfaHAgPD0gMCkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgZ2xvYWJsLmF0dGFjaz0gZmFsc2U7XHJcbiAgICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIGxldCBub2RlMSA9IGNjLmZpbmQoXCJDYW52YXMvYmovZC04L2MtOFwiKTtcclxuICAgICAgICBub2RlMS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgIGxldCBub2RlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmovYi9hXCIpO1xyXG4gICAgICAgIG5vZGUyLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICAgc2VsZi5ub2RlLmFjdGl2ZSA9IGZhbHNlO1xyXG4gICAgICAgIGxldCBub2RlMyA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi90Z2NnXCIpO1xyXG4gICAgICAgIG5vZGUzLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoZ2xvYWJsLm1pbmlvbjFfaHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgIFxyXG4gICAgdGhpcy5tYWluX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgIFxyXG4gICAgIFxyXG4gICAgfVxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBpZiAoZ2xvYWJsLmF0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgIFxyXG4gICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9IHRoaXMubWFpbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSB0aGlzLm1haW5feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBcclxuICAgICAgICB9XHJcblxyXG4gICAgfVxyXG4gICAgXHJcbn1cclxuXHJcblxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/enemy - 003.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'eacdaUcae9GSpLgd0h4kvHb', 'enemy - 003');
// 3.16小游戏/command_TypeScript/level2/enemy - 003.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main_2 = require("./main");
var enemy_3 = /** @class */ (function (_super) {
    __extends(enemy_3, _super);
    function enemy_3() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_3.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_3.prototype.start = function () {
        this.schedule(function () {
            main_2.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_3.prototype.onCollisionEnter = function (other, self) {
        if (main_2.default.minion_attack == true) {
            main_2.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_3.prototype.onCollisionExit = function (other) {
        main_2.default.minion_attack = false;
        if (main_2.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(main_2.default.main_hp, 19);
        }
    };
    enemy_3.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼2");
        if (this.count == 0) {
            if (node1.active == false) {
                this.node.setPosition(this.node.position.x, node1.position.y);
                var node2 = cc.find("Canvas/bj/b/a");
                node2.setContentSize(400, 26);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (main_2.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_3 = __decorate([
        ccclass
    ], enemy_3);
    return enemy_3;
}(cc.Component));
exports.default = enemy_3;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXGVuZW15IC0gMDAzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLCtCQUEyQjtBQUszQjtJQUFxQywyQkFBWTtJQUFqRDtRQUFBLHFFQThFQztRQXpFSSxXQUFLLEdBQUcsQ0FBQyxDQUFDOztJQXlFZixDQUFDO0lBeEVHLHdCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFFM0IsQ0FBQztJQUdELHVCQUFLLEdBQUw7UUFDSSxJQUFJLENBQUMsUUFBUSxDQUFDO1lBQ1YsY0FBTSxDQUFDLGFBQWEsR0FBRyxJQUFJLENBQUM7UUFDaEMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBRUwsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQUVELGtDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUV2QixJQUFHLGNBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxDQUFDO1lBQ3BCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNwRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO1lBQzVDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN2RCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1NBQzlDO0lBR0wsQ0FBQztJQUVELGlDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUlsQixjQUFNLENBQUMsYUFBYSxHQUFHLEtBQUssQ0FBQztRQUM3QixJQUFJLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxjQUFNLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUdELHdCQUFNLEdBQU4sVUFBUSxFQUFFO1FBRU4sSUFBSSxLQUFLLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxvQkFBb0IsQ0FBQyxDQUFDO1FBQzFDLElBQUksSUFBSSxDQUFDLEtBQUssSUFBSSxDQUFDLEVBQUU7WUFDakIsSUFBSSxLQUFLLENBQUMsTUFBTSxJQUFJLEtBQUssRUFBRTtnQkFFdkIsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFHLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2pFLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUUsZUFBZSxDQUFDLENBQUE7Z0JBQ3JDLEtBQUssQ0FBQyxjQUFjLENBQUMsR0FBRyxFQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUM3QixJQUFJLENBQUMsS0FBSyxFQUFFLENBQUM7YUFDaEI7U0FDSjtRQUVELElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7WUFDdkIsSUFBSSxjQUFNLENBQUMsYUFBYSxJQUFLLElBQUksRUFBRTtnQkFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7YUFHaEY7aUJBQU07Z0JBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsRUFBSTtvQkFDdkcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7aUJBQzNFO2FBRUo7U0FDUDtJQUdMLENBQUM7SUE3RWdCLE9BQU87UUFEM0IsT0FBTztPQUNhLE9BQU8sQ0E4RTNCO0lBQUQsY0FBQztDQTlFRCxBQThFQyxDQTlFb0MsRUFBRSxDQUFDLFNBQVMsR0E4RWhEO2tCQTlFb0IsT0FBTyIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIlxyXG4vLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuaW1wb3J0IG1haW5fMSBmcm9tIFwiLi9tYWluXCJcclxuXHJcblxyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgZW5lbXlfMyBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcbiAgIFxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcbiAgICBtaW5pb25feDtcclxuICAgIFxyXG4gICAgIGNvdW50ID0gMDtcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuXHJcbiAgICB9XHJcbiAgICBcclxuXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgICAgdGhpcy5zY2hlZHVsZSgoKSA9PiB7XHJcbiAgICAgICAgICAgIG1haW5fMS5taW5pb25fYXR0YWNrID0gdHJ1ZTtcclxuICAgICAgICB9LDEpO1xyXG5cclxuICAgICAgICB0aGlzLm1pbmlvbl94ID0gdGhpcy5ub2RlLnBvc2l0aW9uLng7XHJcbiAgICB9XHJcbiAgICBcclxuICAgIG9uQ29sbGlzaW9uRW50ZXIob3RoZXIsc2VsZil7XHJcbiAgICAgICBcclxuICAgICAgICBpZihtYWluXzEubWluaW9uX2F0dGFjayA9PSB0cnVlKSB7XHJcbiAgICAgICAgICAgIG1haW5fMS5tYWluX2hwIC09IDU7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi01XCI7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UyID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL3h5L21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICB9XHJcbiAgICAgIFxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICBcclxuICAgICAgICBcclxuICAgICAgXHJcbiAgICAgICBtYWluXzEubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgICAgaWYoIG1haW5fMS5tYWluX2hwIDw9IDApIHtcclxuICAgICAgICBvdGhlci5ub2RlLmFjdGl2ZSA9IGZhbHNlO1xyXG4gICAgICAgIGxldCBsb3NlID0gY2MuZmluZChcIkNhbnZhcy9iai9mYWlsXCIpO1xyXG4gICAgICAgIGxvc2UuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZSggbWFpbl8xLm1haW5faHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuXHJcbiAgICB1cGRhdGUgKGR0KSB7XHJcbiAgICAgIFxyXG4gICAgICAgIGxldCBub2RlMSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi/lsI/prLwyXCIpO1xyXG4gICAgICAgIGlmICh0aGlzLmNvdW50ID09IDApIHtcclxuICAgICAgICAgICAgaWYgKG5vZGUxLmFjdGl2ZSA9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24oICB0aGlzLm5vZGUucG9zaXRpb24ueCAsIG5vZGUxLnBvc2l0aW9uLnkpOyAgIFxyXG4gICAgICAgICAgICAgICAgbGV0IG5vZGUyID0gY2MuZmluZCAoXCJDYW52YXMvYmovYi9hXCIpXHJcbiAgICAgICAgICAgICAgICBub2RlMi5zZXRDb250ZW50U2l6ZSg0MDAsMjYpO1xyXG4gICAgICAgICAgICAgICAgdGhpcy5jb3VudCsrOyAgICAgICAgXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKG5vZGUxLmFjdGl2ZSA9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICBpZiAobWFpbl8xLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSAgdGhpcy5taW5pb25feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPD0gIHRoaXMubWluaW9uX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIFxyXG4gICAgfVxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/left.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '5ceedp2BGlJTLjvk8lDqHNP', 'left');
// 3.16小游戏/command_TypeScript/level2/left.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var main_2 = require("./main");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight_1 = /** @class */ (function (_super) {
    __extends(leftToRight_1, _super);
    function leftToRight_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    leftToRight_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight_1.prototype.onCollisionEnter = function (other, self) {
        cc.log(other.node.name);
        if (main_2.default.attack == true) {
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            if (other.node.name == "小鬼") {
                main_2.default.minion1_hp -= 30;
            }
            else if (other.node.name == "小鬼2") {
                main_2.default.minion2_hp -= 30;
            }
            else if (other.node.name == "小鬼3") {
                main_2.default.minion3_hp -= 30;
            }
        }
        main_2.default.attack = false;
    };
    leftToRight_1.prototype.onCollisionStay = function (other) {
        main_2.default.attack = false;
        main_2.default.minion_attack = false;
    };
    leftToRight_1.prototype.onCollisionExit = function (other) {
        if ((main_2.default.minion1_hp <= 0) && (other.node.name == "小鬼")) {
            other.node.active = false;
        }
        else if ((main_2.default.minion2_hp <= 0) && (other.node.name == "小鬼2")) {
            other.node.active = false;
        }
        else if ((main_2.default.minion3_hp <= 0) && (other.node.name == "小鬼3")) {
            other.node.active = false;
            cc.director.loadScene("fight3");
        }
        else {
            if (other.node.name == "小鬼") {
                other.node.children[0].setContentSize(main_2.default.minion1_hp, 19);
            }
            else if (other.node.name == "小鬼2") {
                other.node.children[0].setContentSize(main_2.default.minion2_hp, 19);
            }
            else if (other.node.name == "小鬼3") {
                other.node.children[0].setContentSize(main_2.default.minion3_hp, 19);
            }
        }
    };
    leftToRight_1.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight_1.prototype.update = function (dt) {
        if (main_2.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], leftToRight_1.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight_1.prototype, "text", void 0);
    leftToRight_1 = __decorate([
        ccclass
    ], leftToRight_1);
    return leftToRight_1;
}(cc.Component));
exports.default = leftToRight_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXGxlZnQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsK0JBQTJCO0FBS3JCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBOEZDO1FBM0ZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUF3RjNCLENBQUM7SUFsRkcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFFeEIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3ZCLElBQUcsY0FBTSxDQUFDLE1BQU0sSUFBSSxJQUFJLEVBQUM7WUFDckIsSUFBSSxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQywrQkFBK0IsQ0FBQyxDQUFDO1lBQ3RELE1BQU0sQ0FBQyxZQUFZLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7WUFDN0MsSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyw2QkFBNkIsQ0FBQyxDQUFDO1lBQ3JELE9BQU8sQ0FBQyxZQUFZLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUM7WUFDNUMsSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLEVBQUU7Z0JBQzNCLGNBQU0sQ0FBQyxVQUFVLElBQUksRUFBRSxDQUFDO2FBQ3hCO2lCQUFPLElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxFQUFFO2dCQUNwQyxjQUFNLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQzthQUN4QjtpQkFBTyxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssRUFBRTtnQkFDcEMsY0FBTSxDQUFDLFVBQVUsSUFBSSxFQUFFLENBQUM7YUFDeEI7U0FDSDtRQUVELGNBQU0sQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO0lBQzFCLENBQUM7SUFHRCx1Q0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFDakIsY0FBTSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7UUFDdEIsY0FBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7SUFDakMsQ0FBQztJQUVELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQU1sQixJQUFHLENBQUMsY0FBTSxDQUFDLFVBQVUsSUFBSSxDQUFDLENBQUMsSUFBRSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLElBQUksQ0FBQyxFQUFFO1lBQ3ZELEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztTQUMxQjthQUFPLElBQUcsQ0FBQyxjQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDaEUsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQzFCO2FBQU0sSUFBRyxDQUFDLGNBQU0sQ0FBQyxVQUFVLElBQUksQ0FBQyxDQUFDLElBQUUsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLENBQUMsRUFBRTtZQUMvRCxLQUFLLENBQUMsSUFBSSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7WUFDMUIsRUFBRSxDQUFDLFFBQVEsQ0FBQyxTQUFTLENBQUMsUUFBUSxDQUFDLENBQUM7U0FDaEM7YUFBTTtZQUNILElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxFQUFFO2dCQUMzQixLQUFLLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsY0FBTSxDQUFDLFVBQVUsRUFBRSxFQUFFLENBQUMsQ0FBQzthQUM3RDtpQkFBTyxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssRUFBRTtnQkFDcEMsS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFDLGNBQU0sQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLENBQUM7YUFDN0Q7aUJBQU8sSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7Z0JBQ3BDLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxjQUFNLENBQUMsVUFBVSxFQUFFLEVBQUUsQ0FBQyxDQUFDO2FBQzdEO1NBQ0o7SUFFSixDQUFDO0lBRUQsNkJBQUssR0FBTDtRQUVBLElBQUksQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO0lBR25DLENBQUM7SUFFRCw4QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUNOLElBQUksY0FBTSxDQUFDLE1BQU0sSUFBSyxJQUFJLEVBQUU7WUFDM0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FHNUU7YUFBTTtZQUNILElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDLEVBQUk7Z0JBQ2pHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzNFO1NBRUo7SUFHTCxDQUFDO0lBekZEO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7Z0RBQ0k7SUFHdkI7UUFEQyxRQUFROytDQUNjO0lBTk4sYUFBYTtRQURqQyxPQUFPO09BQ2EsYUFBYSxDQThGakM7SUFBRCxvQkFBQztDQTlGRCxBQThGQyxDQTlGMEMsRUFBRSxDQUFDLFNBQVMsR0E4RnREO2tCQTlGb0IsYUFBYSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuaW1wb3J0IG1haW5fMSBmcm9tIFwiLi9tYWluXCJcclxuXHJcblxyXG5cclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgbGVmdFRvUmlnaHRfMSBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcbiAgICBtYWluX3g6IG51bWJlcjtcclxuICAgIFxyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3g6IG51bWJlcjtcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF95OiBudW1iZXI7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcblxyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgLy/kuqfnlJ/norDmkp7kvJrosIPnlKhcclxuICAgIG9uQ29sbGlzaW9uRW50ZXIob3RoZXIsc2VsZil7XHJcbiAgICAgIFxyXG4gICAgICAgY2MubG9nKG90aGVyLm5vZGUubmFtZSk7XHJcbiAgICAgICAgaWYobWFpbl8xLmF0dGFjayA9PSB0cnVlKXtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItMzBcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICAgICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLxcIikge1xyXG4gICAgICAgICAgICBtYWluXzEubWluaW9uMV9ocCAtPSAzMDtcclxuICAgICAgICAgICB9IGVsc2UgIGlmKG90aGVyLm5vZGUubmFtZSA9PSBcIuWwj+msvDJcIikge1xyXG4gICAgICAgICAgICBtYWluXzEubWluaW9uMl9ocCAtPSAzMDtcclxuICAgICAgICAgICB9IGVsc2UgIGlmKG90aGVyLm5vZGUubmFtZSA9PSBcIuWwj+msvDNcIikge1xyXG4gICAgICAgICAgICBtYWluXzEubWluaW9uM19ocCAtPSAzMDtcclxuICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgIG1haW5fMS5hdHRhY2sgPSBmYWxzZTtcclxuICAgIH1cclxuXHJcblxyXG4gICAgb25Db2xsaXNpb25TdGF5KG90aGVyKSB7XHJcbiAgICAgICAgbWFpbl8xLmF0dGFjayA9IGZhbHNlO1xyXG4gICAgICAgIG1haW5fMS5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcblxyXG4gICAgICAgXHJcbiAgICAgICBcclxuICAgICBcclxuICAgICAgXHJcbiAgICAgICBpZigobWFpbl8xLm1pbmlvbjFfaHAgPD0gMCkmJihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLxcIikpIHsgICBcclxuICAgICAgICBvdGhlci5ub2RlLmFjdGl2ZSA9IGZhbHNlO1xyXG4gICAgICAgfSAgZWxzZSBpZigobWFpbl8xLm1pbmlvbjJfaHAgPD0gMCkmJihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgIH0gZWxzZSBpZigobWFpbl8xLm1pbmlvbjNfaHAgPD0gMCkmJihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwzXCIpKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBjYy5kaXJlY3Rvci5sb2FkU2NlbmUoXCJmaWdodDNcIik7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgIGlmKG90aGVyLm5vZGUubmFtZSA9PSBcIuWwj+msvFwiKSB7XHJcbiAgICAgICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUobWFpbl8xLm1pbmlvbjFfaHAsIDE5KTtcclxuICAgICAgICAgICB9IGVsc2UgIGlmKG90aGVyLm5vZGUubmFtZSA9PSBcIuWwj+msvDJcIikge1xyXG4gICAgICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKG1haW5fMS5taW5pb24yX2hwLCAxOSk7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwzXCIpIHtcclxuICAgICAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShtYWluXzEubWluaW9uM19ocCwgMTkpO1xyXG4gICAgICAgICAgIH1cclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgIFxyXG4gICAgdGhpcy5tYWluX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgIFxyXG4gICAgIFxyXG4gICAgfVxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBpZiAobWFpbl8xLmF0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgIFxyXG4gICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9IHRoaXMubWFpbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSB0aGlzLm1haW5feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBcclxuICAgICAgICB9XHJcblxyXG4gICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbn1cclxuXHJcblxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/main.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '91522+UAMROjo6f/8WxasmZ', 'main');
// 3.16小游戏/command_TypeScript/level2/main.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main_1 = /** @class */ (function (_super) {
    __extends(main_1, _super);
    function main_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    main_1_1 = main_1;
    main_1.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    main_1.prototype.onEventStart = function () {
        main_1_1.attack = true;
    };
    main_1.prototype.update = function (dt) {
        cc.log("min1, min2, min3 " + main_1_1.minion1_hp + " " + main_1_1.minion3_hp + " " + main_1_1.minion3_hp);
    };
    var main_1_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    main_1.attack = false;
    main_1.minion_attack = false;
    main_1.main_hp = 163;
    main_1.minion1_hp = 163;
    main_1.minion2_hp = 163;
    main_1.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], main_1.prototype, "label", void 0);
    __decorate([
        property
    ], main_1.prototype, "text", void 0);
    main_1 = main_1_1 = __decorate([
        ccclass
    ], main_1);
    return main_1;
}(cc.Component));
exports.default = main_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXG1haW4udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7OztBQUFBLG9CQUFvQjtBQUNwQix3RUFBd0U7QUFDeEUsbUJBQW1CO0FBQ25CLGtGQUFrRjtBQUNsRiw4QkFBOEI7QUFDOUIsa0ZBQWtGOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFFNUUsSUFBQSxLQUFzQixFQUFFLENBQUMsVUFBVSxFQUFsQyxPQUFPLGFBQUEsRUFBRSxRQUFRLGNBQWlCLENBQUM7QUFHMUM7SUFBb0MsMEJBQVk7SUFBaEQ7UUFBQSxxRUE4QkM7UUEzQkcsV0FBSyxHQUFhLElBQUksQ0FBQztRQUd2QixVQUFJLEdBQVcsT0FBTyxDQUFDOztJQXdCM0IsQ0FBQztlQTlCb0IsTUFBTTtJQW9CdkIsc0JBQUssR0FBTDtRQUNJLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLFlBQVksRUFBRSxJQUFJLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3hELENBQUM7SUFDRCw2QkFBWSxHQUFaO1FBQ0ksUUFBTSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7SUFFekIsQ0FBQztJQUNELHVCQUFNLEdBQU4sVUFBUSxFQUFFO1FBQ04sRUFBRSxDQUFDLEdBQUcsQ0FBQyxtQkFBbUIsR0FBQyxRQUFNLENBQUMsVUFBVSxHQUFDLEdBQUcsR0FBQyxRQUFNLENBQUMsVUFBVSxHQUFDLEdBQUcsR0FBQyxRQUFNLENBQUMsVUFBVSxDQUFDLENBQUE7SUFDN0YsQ0FBQzs7SUFyQkQsd0JBQXdCO0lBRXhCLGVBQWU7SUFFRixhQUFNLEdBQVksS0FBSyxDQUFDO0lBQ3hCLG9CQUFhLEdBQVksS0FBSyxDQUFDO0lBQy9CLGNBQU8sR0FBVyxHQUFHLENBQUM7SUFDdEIsaUJBQVUsR0FBVyxHQUFHLENBQUM7SUFDekIsaUJBQVUsR0FBVyxHQUFHLENBQUM7SUFDekIsaUJBQVUsR0FBVyxHQUFHLENBQUM7SUFkdEM7UUFEQyxRQUFRLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQzt5Q0FDSTtJQUd2QjtRQURDLFFBQVE7d0NBQ2M7SUFOTixNQUFNO1FBRDFCLE9BQU87T0FDYSxNQUFNLENBOEIxQjtJQUFELGFBQUM7Q0E5QkQsQUE4QkMsQ0E5Qm1DLEVBQUUsQ0FBQyxTQUFTLEdBOEIvQztrQkE5Qm9CLE1BQU0iLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyIvLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIG1haW5fMSBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcblxyXG4gICAgLy8gb25Mb2FkICgpIHt9XHJcbiAgIFxyXG4gICBwdWJsaWMgc3RhdGljIGF0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbl9hdHRhY2s6IGJvb2xlYW4gPSBmYWxzZTtcclxuICAgcHVibGljIHN0YXRpYyBtYWluX2hwOiBudW1iZXIgPSAxNjM7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMV9ocDogbnVtYmVyID0gMTYzO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjJfaHA6IG51bWJlciA9IDE2MztcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24zX2hwOiBudW1iZXIgPSAxNjM7XHJcbiAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLm5vZGUub24oJ3RvdWNoc3RhcnQnLCB0aGlzLm9uRXZlbnRTdGFydCwgdGhpcyk7XHJcbiAgICB9XHJcbiAgICBvbkV2ZW50U3RhcnQoKSB7XHJcbiAgICAgICAgbWFpbl8xLmF0dGFjayA9IHRydWU7XHJcbiAgICAgICBcclxuICAgIH1cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBjYy5sb2coXCJtaW4xLCBtaW4yLCBtaW4zIFwiK21haW5fMS5taW5pb24xX2hwK1wiIFwiK21haW5fMS5taW5pb24zX2hwK1wiIFwiK21haW5fMS5taW5pb24zX2hwKVxyXG4gICAgfVxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/enemy - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'eb542+xnSBCSoVJvCdOiD17', 'enemy - 001');
// 3.16小游戏/command_TypeScript/level2/enemy - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main_2 = require("./main");
var enemy_1 = /** @class */ (function (_super) {
    __extends(enemy_1, _super);
    function enemy_1() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    enemy_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_1.prototype.start = function () {
        this.schedule(function () {
            main_2.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_1.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (main_2.default.minion_attack == true) {
            main_2.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_1.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        main_2.default.minion_attack = false;
        if (main_2.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(main_2.default.main_hp, 19);
        }
    };
    enemy_1.prototype.update = function (dt) {
        if (main_2.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    enemy_1 = __decorate([
        ccclass
    ], enemy_1);
    return enemy_1;
}(cc.Component));
exports.default = enemy_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXGVuZW15IC0gMDAxLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLCtCQUEyQjtBQUszQjtJQUFxQywyQkFBWTtJQUFqRDs7SUE0REEsQ0FBQztJQXZERyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBQzNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLGNBQU0sQ0FBQyxhQUFhLEdBQUcsSUFBSSxDQUFDO1FBQ2hDLENBQUMsRUFBQyxDQUFDLENBQUMsQ0FBQztRQUVMLElBQUksQ0FBQyxRQUFRLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUM7SUFFRCxrQ0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFDdkIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ3pCLElBQUcsY0FBTSxDQUFDLGFBQWEsSUFBSSxJQUFJLEVBQUU7WUFDN0IsY0FBTSxDQUFDLE9BQU8sSUFBSSxDQUFDLENBQUM7WUFDcEIsSUFBSSxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyw2QkFBNkIsQ0FBQyxDQUFDO1lBQ3BELE1BQU0sQ0FBQyxZQUFZLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7WUFDNUMsSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQywrQkFBK0IsQ0FBQyxDQUFDO1lBQ3ZELE9BQU8sQ0FBQyxZQUFZLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUM7U0FDOUM7SUFFTCxDQUFDO0lBRUQsaUNBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2xCLEVBQUUsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7UUFHZixjQUFNLENBQUMsYUFBYSxHQUFHLEtBQUssQ0FBQztRQUM3QixJQUFJLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxjQUFNLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUdELHdCQUFNLEdBQU4sVUFBUSxFQUFFO1FBRU4sSUFBSSxjQUFNLENBQUMsYUFBYSxJQUFLLElBQUksRUFBRTtZQUMvQixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUksSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUUsQ0FBQztTQUdoRjthQUFNO1lBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsRUFBSTtnQkFDdkcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDM0U7U0FFSjtJQUNSLENBQUM7SUEzRGdCLE9BQU87UUFEM0IsT0FBTztPQUNhLE9BQU8sQ0E0RDNCO0lBQUQsY0FBQztDQTVERCxBQTREQyxDQTVEb0MsRUFBRSxDQUFDLFNBQVMsR0E0RGhEO2tCQTVEb0IsT0FBTyIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIlxyXG4vLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuaW1wb3J0IG1haW5fMSBmcm9tIFwiLi9tYWluXCJcclxuaW1wb3J0IGxlZnRUb1JpZ2h0IGZyb20gXCIuL2xlZnRcIlxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgbWFpbl8xLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgIGlmKG1haW5fMS5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgbWFpbl8xLm1haW5faHAgLT0gNTtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTVcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgICAgY2MubG9nKFwi56Kw5pKe57uT5p2fXCIpO1xyXG4gICAgICAgIFxyXG4gICAgICBcclxuICAgICAgIG1haW5fMS5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggbWFpbl8xLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBtYWluXzEubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuXHJcbiAgICAgICAgaWYgKG1haW5fMS5taW5pb25fYXR0YWNrID09ICB0cnVlKSB7XHJcbiAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIFxyXG4gICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54ID49ICB0aGlzLm1pbmlvbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSAgdGhpcy5taW5pb25feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCArIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICBcclxuICAgICAgICAgICB9XHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level3/gloabl.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '78d89LIGApLnLh0BvM0+HHz', 'gloabl');
// 3.16小游戏/command_TypeScript/level3/gloabl.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var gloabl = /** @class */ (function (_super) {
    __extends(gloabl, _super);
    function gloabl() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    gloabl_1 = gloabl;
    gloabl.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    gloabl.prototype.onEventStart = function () {
        cc.log("click");
        gloabl_1.attack = true;
    };
    var gloabl_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    gloabl.attack = false;
    gloabl.minion_attack = false;
    gloabl.main_hp = 163;
    gloabl.minion1_hp = 280;
    gloabl.minion2_hp = 163;
    gloabl.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], gloabl.prototype, "label", void 0);
    __decorate([
        property
    ], gloabl.prototype, "text", void 0);
    gloabl = gloabl_1 = __decorate([
        ccclass
    ], gloabl);
    return gloabl;
}(cc.Component));
exports.default = gloabl;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDNcXGdsb2FibC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUFvQywwQkFBWTtJQUFoRDtRQUFBLHFFQTRCQztRQXpCRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBc0IzQixDQUFDO2VBNUJvQixNQUFNO0lBb0J2QixzQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEQsQ0FBQztJQUNELDZCQUFZLEdBQVo7UUFDSSxFQUFFLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ2hCLFFBQU0sQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO0lBQ3pCLENBQUM7O0lBbEJELHdCQUF3QjtJQUV4QixlQUFlO0lBRUYsYUFBTSxHQUFZLEtBQUssQ0FBQztJQUN4QixvQkFBYSxHQUFZLEtBQUssQ0FBQztJQUMvQixjQUFPLEdBQVcsR0FBRyxDQUFDO0lBQ3RCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBZHRDO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7eUNBQ0k7SUFHdkI7UUFEQyxRQUFRO3dDQUNjO0lBTk4sTUFBTTtRQUQxQixPQUFPO09BQ2EsTUFBTSxDQTRCMUI7SUFBRCxhQUFDO0NBNUJELEFBNEJDLENBNUJtQyxFQUFFLENBQUMsU0FBUyxHQTRCL0M7a0JBNUJvQixNQUFNIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBnbG9hYmwgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG5cclxuICAgIEBwcm9wZXJ0eShjYy5MYWJlbClcclxuICAgIGxhYmVsOiBjYy5MYWJlbCA9IG51bGw7XHJcblxyXG4gICAgQHByb3BlcnR5XHJcbiAgICB0ZXh0OiBzdHJpbmcgPSAnaGVsbG8nO1xyXG5cclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgXHJcbiAgICAvLyBvbkxvYWQgKCkge31cclxuICAgXHJcbiAgIHB1YmxpYyBzdGF0aWMgYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uX2F0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1haW5faHA6IG51bWJlciA9IDE2MztcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24xX2hwOiBudW1iZXIgPSAyODA7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMl9ocDogbnVtYmVyID0gMTYzO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjNfaHA6IG51bWJlciA9IDE2MztcclxuICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMubm9kZS5vbigndG91Y2hzdGFydCcsIHRoaXMub25FdmVudFN0YXJ0LCB0aGlzKTtcclxuICAgIH1cclxuICAgIG9uRXZlbnRTdGFydCgpIHtcclxuICAgICAgICBjYy5sb2coXCJjbGlja1wiKTtcclxuICAgICAgICBnbG9hYmwuYXR0YWNrID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level3/right.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '7fff0erarVM4LpvOjguWiEk', 'right');
// 3.16小游戏/command_TypeScript/level3/right.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var gloabl_1 = require("./gloabl");
var right = /** @class */ (function (_super) {
    __extends(right, _super);
    function right() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    right.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    right.prototype.start = function () {
        this.schedule(function () {
            gloabl_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    right.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (gloabl_1.default.minion_attack == true) {
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-20";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
            gloabl_1.default.main_hp -= 20;
        }
    };
    right.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        gloabl_1.default.minion_attack = false;
        if (gloabl_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(gloabl_1.default.main_hp, 19);
        }
    };
    right.prototype.update = function (dt) {
        if (gloabl_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    right = __decorate([
        ccclass
    ], right);
    return right;
}(cc.Component));
exports.default = right;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDNcXHJpZ2h0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLG1DQUE2QjtBQUs3QjtJQUFtQyx5QkFBWTtJQUEvQzs7SUE2REEsQ0FBQztJQXhERyxzQkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBQzNCLENBQUM7SUFHRCxxQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLGdCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsZ0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBQ3ZCLEVBQUUsQ0FBQyxHQUFHLENBQUMsTUFBTSxHQUFDLEtBQUssQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUN6QixJQUFHLGdCQUFNLENBQUMsYUFBYSxJQUFJLElBQUksRUFBRTtZQUM3QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUM3QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztZQUMzQyxnQkFBTSxDQUFDLE9BQU8sSUFBSSxFQUFFLENBQUM7U0FDeEI7SUFHTCxDQUFDO0lBRUQsK0JBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2xCLEVBQUUsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7UUFHZixnQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7UUFDN0IsSUFBSSxnQkFBTSxDQUFDLE9BQU8sSUFBSSxDQUFDLEVBQUU7WUFDeEIsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLElBQUksSUFBSSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsZ0JBQWdCLENBQUMsQ0FBQztZQUNyQyxJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztTQUNuQjthQUFNO1lBQ04sS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFFLGdCQUFNLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUdELHNCQUFNLEdBQU4sVUFBUSxFQUFFO1FBRU4sSUFBSSxnQkFBTSxDQUFDLGFBQWEsSUFBSyxJQUFJLEVBQUU7WUFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7U0FHaEY7YUFBTTtZQUNILElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLEVBQUk7Z0JBQ3ZHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzNFO1NBRUo7SUFDUixDQUFDO0lBNURnQixLQUFLO1FBRHpCLE9BQU87T0FDYSxLQUFLLENBNkR6QjtJQUFELFlBQUM7Q0E3REQsQUE2REMsQ0E3RGtDLEVBQUUsQ0FBQyxTQUFTLEdBNkQ5QztrQkE3RG9CLEtBQUsiLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyJcclxuLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcbmltcG9ydCBnbG9hYmwgZnJvbSBcIi4vZ2xvYWJsXCJcclxuaW1wb3J0IG1haW5DaGFyYWN0ZXIgZnJvbSBcIi4vbWFpbiAtIDAwMVwiXHJcblxyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgcmlnaHQgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgIGlmKGdsb2FibC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTIwXCI7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UyID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL3h5L21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICAgICAgZ2xvYWJsLm1haW5faHAgLT0gMjA7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICAgICBjYy5sb2coXCLnorDmkp7nu5PmnZ9cIik7XHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgIGlmKCBnbG9hYmwubWFpbl9ocCA8PSAwKSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBsZXQgbG9zZSA9IGNjLmZpbmQoXCJDYW52YXMvYmovZmFpbFwiKTtcclxuICAgICAgICBsb3NlLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoIGdsb2FibC5tYWluX2hwLCAxOSk7XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG5cclxuICAgICAgICBpZiAoZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgXHJcbiAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gIHRoaXMubWluaW9uX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9ICB0aGlzLm1pbmlvbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIH1cclxuICAgIH1cclxufVxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level4/enemy2.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'b3216Mf/vtC4YnflgAkn1tW', 'enemy2');
// 3.16小游戏/command_TypeScript/level4/enemy2.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global_1 = require("./global");
var enemy_2 = /** @class */ (function (_super) {
    __extends(enemy_2, _super);
    function enemy_2() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_2.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_2.prototype.start = function () {
        this.schedule(function () {
            global_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_2.prototype.onCollisionEnter = function (other, self) {
        if (global_1.default.minion_attack == true) {
            global_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_2.prototype.onCollisionExit = function (other) {
        global_1.default.minion_attack = false;
        if (global_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global_1.default.main_hp, 19);
        }
    };
    enemy_2.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼");
        if (this.count == 0) {
            if (node1.active == false) {
                var node2 = cc.find("Canvas/bj/b/a");
                node2.active = true;
                node2.setContentSize(200, 26);
                this.node.setPosition(this.node.position.x, node1.position.y);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (global_1.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_2 = __decorate([
        ccclass
    ], enemy_2);
    return enemy_2;
}(cc.Component));
exports.default = enemy_2;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDRcXGVuZW15Mi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQyxtQ0FBNkI7QUFLN0I7SUFBcUMsMkJBQVk7SUFBakQ7UUFBQSxxRUE4RUM7UUF6RUksV0FBSyxHQUFHLENBQUMsQ0FBQzs7SUF5RWYsQ0FBQztJQXhFRyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBRTNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLGdCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsa0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBRXZCLElBQUcsZ0JBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLGdCQUFNLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUM1QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztJQUdMLENBQUM7SUFFRCxpQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFJakIsZ0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzlCLElBQUksZ0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxnQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsbUJBQW1CLENBQUMsQ0FBQztRQUN6QyxJQUFJLElBQUksQ0FBQyxLQUFLLElBQUksQ0FBQyxFQUFFO1lBQ2pCLElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7Z0JBQ3ZCLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUUsZUFBZSxDQUFDLENBQUE7Z0JBQ3JDLEtBQUssQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO2dCQUNwQixLQUFLLENBQUMsY0FBYyxDQUFDLEdBQUcsRUFBQyxFQUFFLENBQUMsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFHLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2pFLElBQUksQ0FBQyxLQUFLLEVBQUUsQ0FBQzthQUNoQjtTQUNKO1FBRUQsSUFBSSxLQUFLLENBQUMsTUFBTSxJQUFJLEtBQUssRUFBRTtZQUN2QixJQUFJLGdCQUFNLENBQUMsYUFBYSxJQUFLLElBQUksRUFBRTtnQkFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7YUFHaEY7aUJBQU07Z0JBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsRUFBSTtvQkFDdkcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7aUJBQzNFO2FBRUo7U0FDUDtJQUdMLENBQUM7SUE3RWdCLE9BQU87UUFEM0IsT0FBTztPQUNhLE9BQU8sQ0E4RTNCO0lBQUQsY0FBQztDQTlFRCxBQThFQyxDQTlFb0MsRUFBRSxDQUFDLFNBQVMsR0E4RWhEO2tCQTlFb0IsT0FBTyIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIlxyXG4vLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuaW1wb3J0IGdsb2JhbCBmcm9tIFwiLi9nbG9iYWxcIlxyXG5cclxuXHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBlbmVteV8yIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuICAgXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuICAgIG1pbmlvbl94O1xyXG4gICAgXHJcbiAgICAgY291bnQgPSAwO1xyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG5cclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgIFxyXG4gICAgICAgIGlmKGdsb2JhbC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1haW5faHAgLT0gNTtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTVcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgXHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgIGdsb2JhbC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYmFsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9iYWwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgXHJcbiAgICAgICAgbGV0IG5vZGUxID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL+Wwj+msvFwiKTtcclxuICAgICAgICBpZiAodGhpcy5jb3VudCA9PSAwKSB7XHJcbiAgICAgICAgICAgIGlmIChub2RlMS5hY3RpdmUgPT0gZmFsc2UpIHtcclxuICAgICAgICAgICAgICAgIGxldCBub2RlMiA9IGNjLmZpbmQgKFwiQ2FudmFzL2JqL2IvYVwiKVxyXG4gICAgICAgICAgICAgICAgbm9kZTIuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIG5vZGUyLnNldENvbnRlbnRTaXplKDIwMCwyNik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24oICB0aGlzLm5vZGUucG9zaXRpb24ueCAsIG5vZGUxLnBvc2l0aW9uLnkpOyAgIFxyXG4gICAgICAgICAgICAgICAgdGhpcy5jb3VudCsrOyAgICAgICAgXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKG5vZGUxLmFjdGl2ZSA9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICBpZiAoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSAgdGhpcy5taW5pb25feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPD0gIHRoaXMubWluaW9uX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIFxyXG4gICAgfVxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level4/global.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '3e3beEVXiNJIp1MVCiH/1uQ', 'global');
// 3.16小游戏/command_TypeScript/level4/global.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global = /** @class */ (function (_super) {
    __extends(global, _super);
    function global() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    global_1 = global;
    global.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    global.prototype.onEventStart = function () {
        global_1.attack = true;
    };
    global.prototype.update = function (dt) {
        cc.log("min1, min2, min3 " + global_1.minion1_hp + " " + global_1.minion3_hp + " " + global_1.minion3_hp);
    };
    var global_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    global.attack = false;
    global.minion_attack = false;
    global.main_hp = 163;
    global.minion1_hp = 280;
    global.minion2_hp = 163;
    global.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], global.prototype, "label", void 0);
    __decorate([
        property
    ], global.prototype, "text", void 0);
    global = global_1 = __decorate([
        ccclass
    ], global);
    return global;
}(cc.Component));
exports.default = global;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDRcXGdsb2JhbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUFvQywwQkFBWTtJQUFoRDtRQUFBLHFFQThCQztRQTNCRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBd0IzQixDQUFDO2VBOUJvQixNQUFNO0lBb0J2QixzQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEQsQ0FBQztJQUNELDZCQUFZLEdBQVo7UUFDSSxRQUFNLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztJQUV6QixDQUFDO0lBQ0QsdUJBQU0sR0FBTixVQUFRLEVBQUU7UUFDTixFQUFFLENBQUMsR0FBRyxDQUFDLG1CQUFtQixHQUFDLFFBQU0sQ0FBQyxVQUFVLEdBQUMsR0FBRyxHQUFDLFFBQU0sQ0FBQyxVQUFVLEdBQUMsR0FBRyxHQUFDLFFBQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQTtJQUM3RixDQUFDOztJQXJCRCx3QkFBd0I7SUFFeEIsZUFBZTtJQUVGLGFBQU0sR0FBWSxLQUFLLENBQUM7SUFDeEIsb0JBQWEsR0FBWSxLQUFLLENBQUM7SUFDL0IsY0FBTyxHQUFXLEdBQUcsQ0FBQztJQUN0QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQUN6QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQUN6QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQWR0QztRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO3lDQUNJO0lBR3ZCO1FBREMsUUFBUTt3Q0FDYztJQU5OLE1BQU07UUFEMUIsT0FBTztPQUNhLE1BQU0sQ0E4QjFCO0lBQUQsYUFBQztDQTlCRCxBQThCQyxDQTlCbUMsRUFBRSxDQUFDLFNBQVMsR0E4Qi9DO2tCQTlCb0IsTUFBTSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgZ2xvYmFsIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuXHJcbiAgICBAcHJvcGVydHkoY2MuTGFiZWwpXHJcbiAgICBsYWJlbDogY2MuTGFiZWwgPSBudWxsO1xyXG5cclxuICAgIEBwcm9wZXJ0eVxyXG4gICAgdGV4dDogc3RyaW5nID0gJ2hlbGxvJztcclxuXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuXHJcbiAgICAvLyBvbkxvYWQgKCkge31cclxuICAgXHJcbiAgIHB1YmxpYyBzdGF0aWMgYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uX2F0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1haW5faHA6IG51bWJlciA9IDE2MztcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24xX2hwOiBudW1iZXIgPSAyODA7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMl9ocDogbnVtYmVyID0gMTYzO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjNfaHA6IG51bWJlciA9IDE2MztcclxuICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMubm9kZS5vbigndG91Y2hzdGFydCcsIHRoaXMub25FdmVudFN0YXJ0LCB0aGlzKTtcclxuICAgIH1cclxuICAgIG9uRXZlbnRTdGFydCgpIHtcclxuICAgICAgICBnbG9iYWwuYXR0YWNrID0gdHJ1ZTtcclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgdXBkYXRlIChkdCkge1xyXG4gICAgICAgIGNjLmxvZyhcIm1pbjEsIG1pbjIsIG1pbjMgXCIrZ2xvYmFsLm1pbmlvbjFfaHArXCIgXCIrZ2xvYmFsLm1pbmlvbjNfaHArXCIgXCIrZ2xvYmFsLm1pbmlvbjNfaHApXHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level4/main - 002.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '4fdc0OSg/hIE7QH4s2Hqk/c', 'main - 002');
// 3.16小游戏/command_TypeScript/level4/main - 002.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global_1 = require("./global");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight_1 = /** @class */ (function (_super) {
    __extends(leftToRight_1, _super);
    function leftToRight_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    leftToRight_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight_1.prototype.onCollisionEnter = function (other, self) {
        cc.log(other.node.name);
        if (global_1.default.attack == true) {
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            if (other.node.name == "小鬼") {
                global_1.default.minion1_hp -= 30;
            }
            else if (other.node.name == "小鬼2") {
                global_1.default.minion2_hp -= 30;
            }
            else if (other.node.name == "小鬼3") {
                global_1.default.minion3_hp -= 30;
            }
        }
        global_1.default.attack = false;
    };
    leftToRight_1.prototype.onCollisionStay = function (other) {
        global_1.default.attack = false;
        global_1.default.minion_attack = false;
    };
    leftToRight_1.prototype.onCollisionExit = function (other) {
        if ((global_1.default.minion1_hp <= 0) && (other.node.name == "小鬼")) {
            other.node.active = false;
        }
        else if ((global_1.default.minion2_hp <= 0) && (other.node.name == "小鬼2")) {
            other.node.active = false;
        }
        else if ((global_1.default.minion3_hp <= 0) && (other.node.name == "小鬼3")) {
            other.node.active = false;
            cc.director.loadScene("fight5");
        }
        else {
            if (other.node.name == "小鬼") {
                other.node.children[0].setContentSize(global_1.default.minion1_hp, 19);
            }
            else if (other.node.name == "小鬼2") {
                other.node.children[0].setContentSize(global_1.default.minion2_hp, 19);
            }
            else if (other.node.name == "小鬼3") {
                other.node.children[0].setContentSize(global_1.default.minion3_hp, 19);
            }
        }
    };
    leftToRight_1.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight_1.prototype.update = function (dt) {
        if (global_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], leftToRight_1.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight_1.prototype, "text", void 0);
    leftToRight_1 = __decorate([
        ccclass
    ], leftToRight_1);
    return leftToRight_1;
}(cc.Component));
exports.default = leftToRight_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDRcXG1haW4gLSAwMDIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsbUNBQTZCO0FBS3ZCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBOEZDO1FBM0ZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUF3RjNCLENBQUM7SUFsRkcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFFeEIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3ZCLElBQUcsZ0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN0RCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzdDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNyRCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1lBQzVDLElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxFQUFFO2dCQUMzQixnQkFBTSxDQUFDLFVBQVUsSUFBSSxFQUFFLENBQUM7YUFDeEI7aUJBQU8sSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7Z0JBQ3BDLGdCQUFNLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQzthQUN4QjtpQkFBTyxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssRUFBRTtnQkFDcEMsZ0JBQU0sQ0FBQyxVQUFVLElBQUksRUFBRSxDQUFDO2FBQ3hCO1NBQ0g7UUFFRCxnQkFBTSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7SUFDMUIsQ0FBQztJQUdELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNqQixnQkFBTSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7UUFDdEIsZ0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO0lBQ2pDLENBQUM7SUFFRCx1Q0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFNbEIsSUFBRyxDQUFDLGdCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLEVBQUU7WUFDdkQsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQzFCO2FBQU8sSUFBRyxDQUFDLGdCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDaEUsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQzFCO2FBQU0sSUFBRyxDQUFDLGdCQUFNLENBQUMsVUFBVSxJQUFJLENBQUMsQ0FBQyxJQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDL0QsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLEVBQUUsQ0FBQyxRQUFRLENBQUMsU0FBUyxDQUFDLFFBQVEsQ0FBQyxDQUFDO1NBQ2hDO2FBQU07WUFDSCxJQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLElBQUksRUFBRTtnQkFDM0IsS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFDLGdCQUFNLENBQUMsVUFBVSxFQUFFLEVBQUUsQ0FBQyxDQUFDO2FBQzdEO2lCQUFPLElBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxFQUFFO2dCQUNwQyxLQUFLLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsZ0JBQU0sQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLENBQUM7YUFDN0Q7aUJBQU8sSUFBRyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7Z0JBQ3BDLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxnQkFBTSxDQUFDLFVBQVUsRUFBRSxFQUFFLENBQUMsQ0FBQzthQUM3RDtTQUNKO0lBRUosQ0FBQztJQUVELDZCQUFLLEdBQUw7UUFFQSxJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUduQyxDQUFDO0lBRUQsOEJBQU0sR0FBTixVQUFRLEVBQUU7UUFDTixJQUFJLGdCQUFNLENBQUMsTUFBTSxJQUFLLElBQUksRUFBRTtZQUMzQixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUc1RTthQUFNO1lBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUMsRUFBSTtnQkFDakcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDM0U7U0FFSjtJQUdMLENBQUM7SUF6RkQ7UUFEQyxRQUFRLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQztnREFDSTtJQUd2QjtRQURDLFFBQVE7K0NBQ2M7SUFOTixhQUFhO1FBRGpDLE9BQU87T0FDYSxhQUFhLENBOEZqQztJQUFELG9CQUFDO0NBOUZELEFBOEZDLENBOUYwQyxFQUFFLENBQUMsU0FBUyxHQThGdEQ7a0JBOUZvQixhQUFhIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbFwiXHJcblxyXG5cclxuXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGxlZnRUb1JpZ2h0XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG5cclxuICAgIEBwcm9wZXJ0eShjYy5MYWJlbClcclxuICAgIGxhYmVsOiBjYy5MYWJlbCA9IG51bGw7XHJcblxyXG4gICAgQHByb3BlcnR5XHJcbiAgICB0ZXh0OiBzdHJpbmcgPSAnaGVsbG8nO1xyXG4gICAgbWFpbl94OiBudW1iZXI7XHJcbiAgICBcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF94OiBudW1iZXI7XHJcbiAgICBwdWJsaWMgc3RhdGljIGN1cnJlbnRfeTogbnVtYmVyO1xyXG5cclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG5cclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIC8v5Lqn55Sf56Kw5pKe5Lya6LCD55SoXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICBcclxuICAgICAgIGNjLmxvZyhvdGhlci5ub2RlLm5hbWUpO1xyXG4gICAgICAgIGlmKGdsb2JhbC5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTMwXCI7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UyID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgICAgaWYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8XCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjFfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjJfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwzXCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjNfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBnbG9iYWwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG5cclxuICAgIG9uQ29sbGlzaW9uU3RheShvdGhlcikge1xyXG4gICAgICAgIGdsb2JhbC5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBnbG9iYWwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG5cclxuICAgICAgIFxyXG4gICAgICAgXHJcbiAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgaWYoKGdsb2JhbC5taW5pb24xX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8XCIpKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgIH0gIGVsc2UgaWYoKGdsb2JhbC5taW5pb24yX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8MlwiKSkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICB9IGVsc2UgaWYoKGdsb2JhbC5taW5pb24zX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8M1wiKSkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgY2MuZGlyZWN0b3IubG9hZFNjZW5lKFwiZmlnaHQ1XCIpO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLxcIikge1xyXG4gICAgICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKGdsb2JhbC5taW5pb24xX2hwLCAxOSk7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpIHtcclxuICAgICAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShnbG9iYWwubWluaW9uMl9ocCwgMTkpO1xyXG4gICAgICAgICAgIH0gZWxzZSAgaWYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8M1wiKSB7XHJcbiAgICAgICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoZ2xvYmFsLm1pbmlvbjNfaHAsIDE5KTtcclxuICAgICAgICAgICB9XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuICAgIFxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICBcclxuICAgIHRoaXMubWFpbl94ID0gdGhpcy5ub2RlLnBvc2l0aW9uLng7XHJcbiAgICAgICBcclxuICAgICBcclxuICAgIH1cclxuXHJcbiAgICB1cGRhdGUgKGR0KSB7XHJcbiAgICAgICAgaWYgKGdsb2JhbC5hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICBcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSB0aGlzLm1haW5feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gdGhpcy5tYWluX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggLSAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgfVxyXG5cclxuICAgICBcclxuICAgIH1cclxuICAgIFxyXG59XHJcblxyXG5cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level4/enemy1.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '9707cTcwxhGzaYxTOQYS6Gj', 'enemy1');
// 3.16小游戏/command_TypeScript/level4/enemy1.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global_1 = require("./global");
var enemy_1 = /** @class */ (function (_super) {
    __extends(enemy_1, _super);
    function enemy_1() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    enemy_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_1.prototype.start = function () {
        this.schedule(function () {
            global_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_1.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global_1.default.minion_attack == true) {
            global_1.default.main_hp -= 20;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-20";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_1.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        global_1.default.minion_attack = false;
        if (global_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global_1.default.main_hp, 19);
        }
    };
    enemy_1.prototype.update = function (dt) {
        if (global_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    enemy_1 = __decorate([
        ccclass
    ], enemy_1);
    return enemy_1;
}(cc.Component));
exports.default = enemy_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDRcXGVuZW15MS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQyxtQ0FBNkI7QUFLN0I7SUFBcUMsMkJBQVk7SUFBakQ7O0lBNERBLENBQUM7SUF2REcsd0JBQU0sR0FBTjtRQUNJLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsbUJBQW1CLEVBQUUsQ0FBQztRQUNoRCxPQUFPLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQztJQUMzQixDQUFDO0lBR0QsdUJBQUssR0FBTDtRQUNJLElBQUksQ0FBQyxRQUFRLENBQUM7WUFDVixnQkFBTSxDQUFDLGFBQWEsR0FBRyxJQUFJLENBQUM7UUFDaEMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBRUwsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQUVELGtDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUN2QixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sR0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDekIsSUFBRyxnQkFBTSxDQUFDLGFBQWEsSUFBSSxJQUFJLEVBQUU7WUFDN0IsZ0JBQU0sQ0FBQyxPQUFPLElBQUksRUFBRSxDQUFDO1lBQ3JCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNwRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzdDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN2RCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1NBQzlDO0lBRUwsQ0FBQztJQUVELGlDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNsQixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBR2YsZ0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzdCLElBQUksZ0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxnQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksZ0JBQU0sQ0FBQyxhQUFhLElBQUssSUFBSSxFQUFFO1lBQy9CLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBSSxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBRSxDQUFDO1NBR2hGO2FBQU07WUFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxFQUFJO2dCQUN2RyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMzRTtTQUVKO0lBQ1IsQ0FBQztJQTNEZ0IsT0FBTztRQUQzQixPQUFPO09BQ2EsT0FBTyxDQTREM0I7SUFBRCxjQUFDO0NBNURELEFBNERDLENBNURvQyxFQUFFLENBQUMsU0FBUyxHQTREaEQ7a0JBNURvQixPQUFPIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbFwiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgIGlmKGdsb2JhbC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1haW5faHAgLT0gMjA7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi0yMFwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICAgICBjYy5sb2coXCLnorDmkp7nu5PmnZ9cIik7XHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgIGlmKCBnbG9iYWwubWFpbl9ocCA8PSAwKSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBsZXQgbG9zZSA9IGNjLmZpbmQoXCJDYW52YXMvYmovZmFpbFwiKTtcclxuICAgICAgICBsb3NlLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoIGdsb2JhbC5tYWluX2hwLCAxOSk7XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG5cclxuICAgICAgICBpZiAoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgXHJcbiAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gIHRoaXMubWluaW9uX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9ICB0aGlzLm1pbmlvbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIH1cclxuICAgIH1cclxufVxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level5/right - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '6fd44YQsKFINrQ6OUrlnRuR', 'right - 001');
// 3.16小游戏/command_TypeScript/level5/right - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___001_1 = require("./global - 001");
var right = /** @class */ (function (_super) {
    __extends(right, _super);
    function right() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    right.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    right.prototype.start = function () {
        this.schedule(function () {
            global___001_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    right.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___001_1.default.minion_attack == true) {
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-20";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
            global___001_1.default.main_hp -= 20;
        }
    };
    right.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        global___001_1.default.minion_attack = false;
        if (global___001_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___001_1.default.main_hp, 19);
        }
    };
    right.prototype.update = function (dt) {
        if (global___001_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    right = __decorate([
        ccclass
    ], right);
    return right;
}(cc.Component));
exports.default = right;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDVcXHJpZ2h0IC0gMDAxLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLCtDQUFtQztBQUtuQztJQUFtQyx5QkFBWTtJQUEvQzs7SUE2REEsQ0FBQztJQXhERyxzQkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBQzNCLENBQUM7SUFHRCxxQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLHNCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsZ0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBQ3ZCLEVBQUUsQ0FBQyxHQUFHLENBQUMsTUFBTSxHQUFDLEtBQUssQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUN6QixJQUFHLHNCQUFNLENBQUMsYUFBYSxJQUFJLElBQUksRUFBRTtZQUM3QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUM3QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztZQUMzQyxzQkFBTSxDQUFDLE9BQU8sSUFBSSxFQUFFLENBQUM7U0FDeEI7SUFHTCxDQUFDO0lBRUQsK0JBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2xCLEVBQUUsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7UUFHZixzQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7UUFDN0IsSUFBSSxzQkFBTSxDQUFDLE9BQU8sSUFBSSxDQUFDLEVBQUU7WUFDeEIsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLElBQUksSUFBSSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsZ0JBQWdCLENBQUMsQ0FBQztZQUNyQyxJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztTQUNuQjthQUFNO1lBQ04sS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFFLHNCQUFNLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUdELHNCQUFNLEdBQU4sVUFBUSxFQUFFO1FBRU4sSUFBSSxzQkFBTSxDQUFDLGFBQWEsSUFBSyxJQUFJLEVBQUU7WUFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7U0FHaEY7YUFBTTtZQUNILElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLEVBQUk7Z0JBQ3ZHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzNFO1NBRUo7SUFDUixDQUFDO0lBNURnQixLQUFLO1FBRHpCLE9BQU87T0FDYSxLQUFLLENBNkR6QjtJQUFELFlBQUM7Q0E3REQsQUE2REMsQ0E3RGtDLEVBQUUsQ0FBQyxTQUFTLEdBNkQ5QztrQkE3RG9CLEtBQUsiLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyJcclxuLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcbmltcG9ydCBnbG9hYmwgZnJvbSBcIi4vZ2xvYmFsIC0gMDAxXCJcclxuXHJcblxyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgcmlnaHQgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgIGlmKGdsb2FibC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTIwXCI7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UyID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL3h5L21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICAgICAgZ2xvYWJsLm1haW5faHAgLT0gMjA7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICAgICBjYy5sb2coXCLnorDmkp7nu5PmnZ9cIik7XHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgIGlmKCBnbG9hYmwubWFpbl9ocCA8PSAwKSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBsZXQgbG9zZSA9IGNjLmZpbmQoXCJDYW52YXMvYmovZmFpbFwiKTtcclxuICAgICAgICBsb3NlLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoIGdsb2FibC5tYWluX2hwLCAxOSk7XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG5cclxuICAgICAgICBpZiAoZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgXHJcbiAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gIHRoaXMubWluaW9uX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9ICB0aGlzLm1pbmlvbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIH1cclxuICAgIH1cclxufVxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level5/global - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'ac1f3CNYoBK/rt2tvbz7Ys1', 'global - 001');
// 3.16小游戏/command_TypeScript/level5/global - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var gloabl = /** @class */ (function (_super) {
    __extends(gloabl, _super);
    function gloabl() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    gloabl_1 = gloabl;
    gloabl.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    gloabl.prototype.onEventStart = function () {
        cc.log("click");
        gloabl_1.attack = true;
    };
    var gloabl_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    gloabl.attack = false;
    gloabl.minion_attack = false;
    gloabl.main_hp = 163;
    gloabl.minion1_hp = 280;
    gloabl.minion2_hp = 163;
    gloabl.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], gloabl.prototype, "label", void 0);
    __decorate([
        property
    ], gloabl.prototype, "text", void 0);
    gloabl = gloabl_1 = __decorate([
        ccclass
    ], gloabl);
    return gloabl;
}(cc.Component));
exports.default = gloabl;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDVcXGdsb2JhbCAtIDAwMS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUFvQywwQkFBWTtJQUFoRDtRQUFBLHFFQTRCQztRQXpCRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBc0IzQixDQUFDO2VBNUJvQixNQUFNO0lBb0J2QixzQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEQsQ0FBQztJQUNELDZCQUFZLEdBQVo7UUFDSSxFQUFFLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ2hCLFFBQU0sQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO0lBQ3pCLENBQUM7O0lBbEJELHdCQUF3QjtJQUV4QixlQUFlO0lBRUYsYUFBTSxHQUFZLEtBQUssQ0FBQztJQUN4QixvQkFBYSxHQUFZLEtBQUssQ0FBQztJQUMvQixjQUFPLEdBQVcsR0FBRyxDQUFDO0lBQ3RCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBZHRDO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7eUNBQ0k7SUFHdkI7UUFEQyxRQUFRO3dDQUNjO0lBTk4sTUFBTTtRQUQxQixPQUFPO09BQ2EsTUFBTSxDQTRCMUI7SUFBRCxhQUFDO0NBNUJELEFBNEJDLENBNUJtQyxFQUFFLENBQUMsU0FBUyxHQTRCL0M7a0JBNUJvQixNQUFNIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBnbG9hYmwgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG5cclxuICAgIEBwcm9wZXJ0eShjYy5MYWJlbClcclxuICAgIGxhYmVsOiBjYy5MYWJlbCA9IG51bGw7XHJcblxyXG4gICAgQHByb3BlcnR5XHJcbiAgICB0ZXh0OiBzdHJpbmcgPSAnaGVsbG8nO1xyXG5cclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgXHJcbiAgICAvLyBvbkxvYWQgKCkge31cclxuICAgXHJcbiAgIHB1YmxpYyBzdGF0aWMgYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uX2F0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1haW5faHA6IG51bWJlciA9IDE2MztcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24xX2hwOiBudW1iZXIgPSAyODA7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMl9ocDogbnVtYmVyID0gMTYzO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjNfaHA6IG51bWJlciA9IDE2MztcclxuICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMubm9kZS5vbigndG91Y2hzdGFydCcsIHRoaXMub25FdmVudFN0YXJ0LCB0aGlzKTtcclxuICAgIH1cclxuICAgIG9uRXZlbnRTdGFydCgpIHtcclxuICAgICAgICBjYy5sb2coXCJjbGlja1wiKTtcclxuICAgICAgICBnbG9hYmwuYXR0YWNrID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level6/right1.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '06a79I1MFlKZYzGvjTIzIZ3', 'right1');
// 3.16小游戏/command_TypeScript/level6/right1.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___002_1 = require("./global - 002");
var enemy_1 = /** @class */ (function (_super) {
    __extends(enemy_1, _super);
    function enemy_1() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    enemy_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_1.prototype.start = function () {
        this.schedule(function () {
            global___002_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_1.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___002_1.default.minion_attack == true) {
            global___002_1.default.main_hp -= 20;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-20";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_1.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        global___002_1.default.minion_attack = false;
        if (global___002_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___002_1.default.main_hp, 19);
        }
    };
    enemy_1.prototype.update = function (dt) {
        if (global___002_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    enemy_1 = __decorate([
        ccclass
    ], enemy_1);
    return enemy_1;
}(cc.Component));
exports.default = enemy_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDZcXHJpZ2h0MS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQywrQ0FBbUM7QUFLbkM7SUFBcUMsMkJBQVk7SUFBakQ7O0lBNERBLENBQUM7SUF2REcsd0JBQU0sR0FBTjtRQUNJLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsbUJBQW1CLEVBQUUsQ0FBQztRQUNoRCxPQUFPLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQztJQUMzQixDQUFDO0lBR0QsdUJBQUssR0FBTDtRQUNJLElBQUksQ0FBQyxRQUFRLENBQUM7WUFDVixzQkFBTSxDQUFDLGFBQWEsR0FBRyxJQUFJLENBQUM7UUFDaEMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBRUwsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQUVELGtDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUN2QixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sR0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDekIsSUFBRyxzQkFBTSxDQUFDLGFBQWEsSUFBSSxJQUFJLEVBQUU7WUFDN0Isc0JBQU0sQ0FBQyxPQUFPLElBQUksRUFBRSxDQUFDO1lBQ3JCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNwRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzdDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN2RCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1NBQzlDO0lBRUwsQ0FBQztJQUVELGlDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNsQixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBR2Ysc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzdCLElBQUksc0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxzQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksc0JBQU0sQ0FBQyxhQUFhLElBQUssSUFBSSxFQUFFO1lBQy9CLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBSSxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBRSxDQUFDO1NBR2hGO2FBQU07WUFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxFQUFJO2dCQUN2RyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMzRTtTQUVKO0lBQ1IsQ0FBQztJQTNEZ0IsT0FBTztRQUQzQixPQUFPO09BQ2EsT0FBTyxDQTREM0I7SUFBRCxjQUFDO0NBNURELEFBNERDLENBNURvQyxFQUFFLENBQUMsU0FBUyxHQTREaEQ7a0JBNURvQixPQUFPIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbCAtIDAwMlwiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgIGlmKGdsb2JhbC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1haW5faHAgLT0gMjA7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi0yMFwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICAgICBjYy5sb2coXCLnorDmkp7nu5PmnZ9cIik7XHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgIGlmKCBnbG9iYWwubWFpbl9ocCA8PSAwKSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBsZXQgbG9zZSA9IGNjLmZpbmQoXCJDYW52YXMvYmovZmFpbFwiKTtcclxuICAgICAgICBsb3NlLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoIGdsb2JhbC5tYWluX2hwLCAxOSk7XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG5cclxuICAgICAgICBpZiAoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgXHJcbiAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gIHRoaXMubWluaW9uX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9ICB0aGlzLm1pbmlvbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIH1cclxuICAgIH1cclxufVxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level6/global - 002.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '895e44MoOdMzJ2X1sOV6aHH', 'global - 002');
// 3.16小游戏/command_TypeScript/level6/global - 002.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global = /** @class */ (function (_super) {
    __extends(global, _super);
    function global() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    global_1 = global;
    global.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    global.prototype.onEventStart = function () {
        global_1.attack = true;
    };
    global.prototype.update = function (dt) {
        cc.log("min1, min2, min3 " + global_1.minion1_hp + " " + global_1.minion3_hp + " " + global_1.minion3_hp);
    };
    var global_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    global.attack = false;
    global.minion_attack = false;
    global.main_hp = 200;
    global.minion1_hp = 400;
    global.minion2_hp = 200;
    global.minion3_hp = 200;
    __decorate([
        property(cc.Label)
    ], global.prototype, "label", void 0);
    __decorate([
        property
    ], global.prototype, "text", void 0);
    global = global_1 = __decorate([
        ccclass
    ], global);
    return global;
}(cc.Component));
exports.default = global;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDZcXGdsb2JhbCAtIDAwMi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUFvQywwQkFBWTtJQUFoRDtRQUFBLHFFQThCQztRQTNCRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBd0IzQixDQUFDO2VBOUJvQixNQUFNO0lBb0J2QixzQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEQsQ0FBQztJQUNELDZCQUFZLEdBQVo7UUFDSSxRQUFNLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztJQUV6QixDQUFDO0lBQ0QsdUJBQU0sR0FBTixVQUFRLEVBQUU7UUFDTixFQUFFLENBQUMsR0FBRyxDQUFDLG1CQUFtQixHQUFDLFFBQU0sQ0FBQyxVQUFVLEdBQUMsR0FBRyxHQUFDLFFBQU0sQ0FBQyxVQUFVLEdBQUMsR0FBRyxHQUFDLFFBQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQTtJQUM3RixDQUFDOztJQXJCRCx3QkFBd0I7SUFFeEIsZUFBZTtJQUVGLGFBQU0sR0FBWSxLQUFLLENBQUM7SUFDeEIsb0JBQWEsR0FBWSxLQUFLLENBQUM7SUFDL0IsY0FBTyxHQUFXLEdBQUcsQ0FBQztJQUN0QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQUN6QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQUN6QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQWR0QztRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO3lDQUNJO0lBR3ZCO1FBREMsUUFBUTt3Q0FDYztJQU5OLE1BQU07UUFEMUIsT0FBTztPQUNhLE1BQU0sQ0E4QjFCO0lBQUQsYUFBQztDQTlCRCxBQThCQyxDQTlCbUMsRUFBRSxDQUFDLFNBQVMsR0E4Qi9DO2tCQTlCb0IsTUFBTSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgZ2xvYmFsIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuXHJcbiAgICBAcHJvcGVydHkoY2MuTGFiZWwpXHJcbiAgICBsYWJlbDogY2MuTGFiZWwgPSBudWxsO1xyXG5cclxuICAgIEBwcm9wZXJ0eVxyXG4gICAgdGV4dDogc3RyaW5nID0gJ2hlbGxvJztcclxuXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuXHJcbiAgICAvLyBvbkxvYWQgKCkge31cclxuICAgXHJcbiAgIHB1YmxpYyBzdGF0aWMgYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uX2F0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1haW5faHA6IG51bWJlciA9IDIwMDtcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24xX2hwOiBudW1iZXIgPSA0MDA7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMl9ocDogbnVtYmVyID0gMjAwO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjNfaHA6IG51bWJlciA9IDIwMDtcclxuICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMubm9kZS5vbigndG91Y2hzdGFydCcsIHRoaXMub25FdmVudFN0YXJ0LCB0aGlzKTtcclxuICAgIH1cclxuICAgIG9uRXZlbnRTdGFydCgpIHtcclxuICAgICAgICBnbG9iYWwuYXR0YWNrID0gdHJ1ZTtcclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgdXBkYXRlIChkdCkge1xyXG4gICAgICAgIGNjLmxvZyhcIm1pbjEsIG1pbjIsIG1pbjMgXCIrZ2xvYmFsLm1pbmlvbjFfaHArXCIgXCIrZ2xvYmFsLm1pbmlvbjNfaHArXCIgXCIrZ2xvYmFsLm1pbmlvbjNfaHApXHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level6/right2.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '673860kFO1O8LmuF5X5RH0G', 'right2');
// 3.16小游戏/command_TypeScript/level6/right2.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___002_1 = require("./global - 002");
var enemy_2 = /** @class */ (function (_super) {
    __extends(enemy_2, _super);
    function enemy_2() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_2.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_2.prototype.start = function () {
        this.schedule(function () {
            global___002_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_2.prototype.onCollisionEnter = function (other, self) {
        if (global___002_1.default.minion_attack == true) {
            global___002_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_2.prototype.onCollisionExit = function (other) {
        global___002_1.default.minion_attack = false;
        if (global___002_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___002_1.default.main_hp, 19);
        }
    };
    enemy_2.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼");
        if (this.count == 0) {
            if (node1.active == false) {
                var node2 = cc.find("Canvas/bj/b/a");
                node2.active = true;
                node2.setContentSize(200, 26);
                this.node.setPosition(this.node.position.x, node1.position.y);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (global___002_1.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_2 = __decorate([
        ccclass
    ], enemy_2);
    return enemy_2;
}(cc.Component));
exports.default = enemy_2;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDZcXHJpZ2h0Mi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQywrQ0FBbUM7QUFLbkM7SUFBcUMsMkJBQVk7SUFBakQ7UUFBQSxxRUE4RUM7UUF6RUksV0FBSyxHQUFHLENBQUMsQ0FBQzs7SUF5RWYsQ0FBQztJQXhFRyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBRTNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLHNCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsa0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBRXZCLElBQUcsc0JBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLHNCQUFNLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUM1QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztJQUdMLENBQUM7SUFFRCxpQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFJakIsc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzlCLElBQUksc0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxzQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsbUJBQW1CLENBQUMsQ0FBQztRQUN6QyxJQUFJLElBQUksQ0FBQyxLQUFLLElBQUksQ0FBQyxFQUFFO1lBQ2pCLElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7Z0JBQ3ZCLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUUsZUFBZSxDQUFDLENBQUE7Z0JBQ3JDLEtBQUssQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO2dCQUNwQixLQUFLLENBQUMsY0FBYyxDQUFDLEdBQUcsRUFBQyxFQUFFLENBQUMsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFHLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2pFLElBQUksQ0FBQyxLQUFLLEVBQUUsQ0FBQzthQUNoQjtTQUNKO1FBRUQsSUFBSSxLQUFLLENBQUMsTUFBTSxJQUFJLEtBQUssRUFBRTtZQUN2QixJQUFJLHNCQUFNLENBQUMsYUFBYSxJQUFLLElBQUksRUFBRTtnQkFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7YUFHaEY7aUJBQU07Z0JBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsRUFBSTtvQkFDdkcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7aUJBQzNFO2FBRUo7U0FDUDtJQUdMLENBQUM7SUE3RWdCLE9BQU87UUFEM0IsT0FBTztPQUNhLE9BQU8sQ0E4RTNCO0lBQUQsY0FBQztDQTlFRCxBQThFQyxDQTlFb0MsRUFBRSxDQUFDLFNBQVMsR0E4RWhEO2tCQTlFb0IsT0FBTyIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIlxyXG4vLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuaW1wb3J0IGdsb2JhbCBmcm9tIFwiLi9nbG9iYWwgLSAwMDJcIlxyXG5cclxuXHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBlbmVteV8yIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuICAgXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuICAgIG1pbmlvbl94O1xyXG4gICAgXHJcbiAgICAgY291bnQgPSAwO1xyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG5cclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgIFxyXG4gICAgICAgIGlmKGdsb2JhbC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1haW5faHAgLT0gNTtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTVcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgXHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgIGdsb2JhbC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYmFsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9iYWwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgXHJcbiAgICAgICAgbGV0IG5vZGUxID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL+Wwj+msvFwiKTtcclxuICAgICAgICBpZiAodGhpcy5jb3VudCA9PSAwKSB7XHJcbiAgICAgICAgICAgIGlmIChub2RlMS5hY3RpdmUgPT0gZmFsc2UpIHtcclxuICAgICAgICAgICAgICAgIGxldCBub2RlMiA9IGNjLmZpbmQgKFwiQ2FudmFzL2JqL2IvYVwiKVxyXG4gICAgICAgICAgICAgICAgbm9kZTIuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIG5vZGUyLnNldENvbnRlbnRTaXplKDIwMCwyNik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24oICB0aGlzLm5vZGUucG9zaXRpb24ueCAsIG5vZGUxLnBvc2l0aW9uLnkpOyAgIFxyXG4gICAgICAgICAgICAgICAgdGhpcy5jb3VudCsrOyAgICAgICAgXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKG5vZGUxLmFjdGl2ZSA9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICBpZiAoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSAgdGhpcy5taW5pb25feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPD0gIHRoaXMubWluaW9uX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIFxyXG4gICAgfVxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level1/leftToRight.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '52d87+5JwxJVpcp4Szvn0th', 'leftToRight');
// 3.16小游戏/command_TypeScript/level1/leftToRight.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var mian_1 = require("./mian");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight = /** @class */ (function (_super) {
    __extends(leftToRight, _super);
    function leftToRight() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    leftToRight_1 = leftToRight;
    // LIFE-CYCLE CALLBACKS:
    leftToRight.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (mian_1.default.attack == true) {
            mian_1.default.minion1_hp -= 30;
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        mian_1.default.attack = false;
    };
    leftToRight.prototype.onCollisionStay = function (other) {
        mian_1.default.attack = false;
        mian_1.default.minion_attack = false;
    };
    leftToRight.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        if (mian_1.default.minion1_hp <= 0) {
            other.node.active = false;
            mian_1.default.attack = false;
            mian_1.default.minion_attack = false;
            mian_1.default.main_hp = 163;
            mian_1.default.minion1_hp = 163;
            mian_1.default.minion2_hp = 163;
            mian_1.default.minion3_hp = 163;
            cc.director.loadScene("fight2");
        }
        else {
            other.node.children[0].setContentSize(mian_1.default.minion1_hp, 19);
        }
    };
    leftToRight.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight.prototype.update = function (dt) {
        if (mian_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
        leftToRight_1.current_x = this.node.position.x;
        leftToRight_1.current_y = this.node.position.y;
    };
    var leftToRight_1;
    __decorate([
        property(cc.Label)
    ], leftToRight.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight.prototype, "text", void 0);
    leftToRight = leftToRight_1 = __decorate([
        ccclass
    ], leftToRight);
    return leftToRight;
}(cc.Component));
exports.default = leftToRight;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDFcXGxlZnRUb1JpZ2h0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBLG9CQUFvQjtBQUNwQix3RUFBd0U7QUFDeEUsbUJBQW1CO0FBQ25CLGtGQUFrRjtBQUNsRiw4QkFBOEI7QUFDOUIsa0ZBQWtGO0FBQ2xGLCtCQUF5QjtBQUtuQixJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUF5QywrQkFBWTtJQUFyRDtRQUFBLHFFQXNGQztRQW5GRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBZ0YzQixDQUFDO29CQXRGb0IsV0FBVztJQVk1Qix3QkFBd0I7SUFFeEIsNEJBQU0sR0FBTjtRQUNJLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsbUJBQW1CLEVBQUUsQ0FBQztRQUNoRCxPQUFPLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQztJQUMzQixDQUFDO0lBR0QsU0FBUztJQUNULHNDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUN2QixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sR0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLENBQUM7UUFFekIsSUFBRyxjQUFJLENBQUMsTUFBTSxJQUFJLElBQUksRUFBQztZQUNuQixjQUFJLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQztZQUN0QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUM3QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDckQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztRQUVELGNBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO0lBRXhCLENBQUM7SUFHRCxxQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFDakIsY0FBSSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7UUFDcEIsY0FBSSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7SUFDL0IsQ0FBQztJQUVELHFDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNsQixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBS2YsSUFBRyxjQUFJLENBQUMsVUFBVSxJQUFJLENBQUMsRUFBRTtZQUN4QixLQUFLLENBQUMsSUFBSSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7WUFDMUIsY0FBSSxDQUFDLE1BQU0sR0FBRSxLQUFLLENBQUM7WUFDbkIsY0FBSSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7WUFDM0IsY0FBSSxDQUFDLE9BQU8sR0FBRSxHQUFHLENBQUM7WUFDbEIsY0FBSSxDQUFDLFVBQVUsR0FBRSxHQUFHLENBQUM7WUFDckIsY0FBSSxDQUFDLFVBQVUsR0FBRSxHQUFHLENBQUM7WUFDckIsY0FBSSxDQUFDLFVBQVUsR0FBRyxHQUFHLENBQUM7WUFDdEIsRUFBRSxDQUFDLFFBQVEsQ0FBQyxTQUFTLENBQUMsUUFBUSxDQUFDLENBQUM7U0FDaEM7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxjQUFJLENBQUMsVUFBVSxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUVELDJCQUFLLEdBQUw7UUFFQSxJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUduQyxDQUFDO0lBRUQsNEJBQU0sR0FBTixVQUFRLEVBQUU7UUFDTixJQUFJLGNBQUksQ0FBQyxNQUFNLElBQUssSUFBSSxFQUFFO1lBQ3pCLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBRzVFO2FBQU07WUFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQyxFQUFJO2dCQUNqRyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMzRTtTQUVKO1FBRUQsYUFBVyxDQUFDLFNBQVMsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7UUFDN0MsYUFBVyxDQUFDLFNBQVMsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDakQsQ0FBQzs7SUFqRkQ7UUFEQyxRQUFRLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQzs4Q0FDSTtJQUd2QjtRQURDLFFBQVE7NkNBQ2M7SUFOTixXQUFXO1FBRC9CLE9BQU87T0FDYSxXQUFXLENBc0YvQjtJQUFELGtCQUFDO0NBdEZELEFBc0ZDLENBdEZ3QyxFQUFFLENBQUMsU0FBUyxHQXNGcEQ7a0JBdEZvQixXQUFXIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5pbXBvcnQgbWFpbiBmcm9tIFwiLi9taWFuXCJcclxuaW1wb3J0IGVuZW15IGZyb20gJy4vZW5lbXknO1xyXG5cclxuXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGxlZnRUb1JpZ2h0IGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuXHJcbiAgICBAcHJvcGVydHkoY2MuTGFiZWwpXHJcbiAgICBsYWJlbDogY2MuTGFiZWwgPSBudWxsO1xyXG5cclxuICAgIEBwcm9wZXJ0eVxyXG4gICAgdGV4dDogc3RyaW5nID0gJ2hlbGxvJztcclxuICAgIG1haW5feDogbnVtYmVyO1xyXG4gICAgXHJcbiAgICBwdWJsaWMgc3RhdGljIGN1cnJlbnRfeDogbnVtYmVyO1xyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3k6IG51bWJlcjtcclxuXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuXHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcbiAgICB9XHJcbiAgICBcclxuXHJcbiAgICAvL+S6p+eUn+eisOaSnuS8muiwg+eUqFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYobWFpbi5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIG1haW4ubWluaW9uMV9ocCAtPSAzMDtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItMzBcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgbWFpbi5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgb25Db2xsaXNpb25TdGF5KG90aGVyKSB7XHJcbiAgICAgICAgbWFpbi5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBtYWluLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgIH1cclxuXHJcbiAgICBvbkNvbGxpc2lvbkV4aXQob3RoZXIpIHtcclxuICAgICAgIGNjLmxvZyhcIueisOaSnue7k+adn1wiKTtcclxuICAgICAgIFxyXG4gICAgICAgXHJcbiAgICAgIFxyXG4gICAgICBcclxuICAgICAgIGlmKG1haW4ubWluaW9uMV9ocCA8PSAwKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBtYWluLmF0dGFjaz0gZmFsc2U7XHJcbiAgICAgICAgbWFpbi5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICAgbWFpbi5tYWluX2hwPSAxNjM7XHJcbiAgICAgICAgbWFpbi5taW5pb24xX2hwPSAxNjM7XHJcbiAgICAgICAgbWFpbi5taW5pb24yX2hwPSAxNjM7XHJcbiAgICAgICAgbWFpbi5taW5pb24zX2hwID0gMTYzO1xyXG4gICAgICAgIGNjLmRpcmVjdG9yLmxvYWRTY2VuZShcImZpZ2h0MlwiKTtcclxuICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShtYWluLm1pbmlvbjFfaHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgIFxyXG4gICAgdGhpcy5tYWluX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgIFxyXG4gICAgIFxyXG4gICAgfVxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBpZiAobWFpbi5hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICBcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSB0aGlzLm1haW5feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gdGhpcy5tYWluX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggLSAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBsZWZ0VG9SaWdodC5jdXJyZW50X3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgICBsZWZ0VG9SaWdodC5jdXJyZW50X3kgPSB0aGlzLm5vZGUucG9zaXRpb24ueTtcclxuICAgIH1cclxuICAgIFxyXG59XHJcblxyXG5cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level6/right3.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'a6149l3q6BDPLdIOQv89K5a', 'right3');
// 3.16小游戏/command_TypeScript/level6/right3.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___002_1 = require("./global - 002");
var enemy_3 = /** @class */ (function (_super) {
    __extends(enemy_3, _super);
    function enemy_3() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_3.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_3.prototype.start = function () {
        this.schedule(function () {
            global___002_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_3.prototype.onCollisionEnter = function (other, self) {
        if (global___002_1.default.minion_attack == true) {
            global___002_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_3.prototype.onCollisionExit = function (other) {
        global___002_1.default.minion_attack = false;
        if (global___002_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___002_1.default.main_hp, 19);
        }
    };
    enemy_3.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼2");
        if (this.count == 0) {
            if (node1.active == false) {
                this.node.setPosition(this.node.position.x, node1.position.y);
                var node2 = cc.find("Canvas/bj/b/a");
                node2.setContentSize(400, 26);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (global___002_1.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_3 = __decorate([
        ccclass
    ], enemy_3);
    return enemy_3;
}(cc.Component));
exports.default = enemy_3;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDZcXHJpZ2h0My50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQywrQ0FBbUM7QUFLbkM7SUFBcUMsMkJBQVk7SUFBakQ7UUFBQSxxRUE4RUM7UUF6RUksV0FBSyxHQUFHLENBQUMsQ0FBQzs7SUF5RWYsQ0FBQztJQXhFRyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBRTNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLHNCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsa0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBRXZCLElBQUcsc0JBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLHNCQUFNLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUM1QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztJQUdMLENBQUM7SUFFRCxpQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFJakIsc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzlCLElBQUksc0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxzQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsb0JBQW9CLENBQUMsQ0FBQztRQUMxQyxJQUFJLElBQUksQ0FBQyxLQUFLLElBQUksQ0FBQyxFQUFFO1lBQ2pCLElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7Z0JBRXZCLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsRUFBRyxLQUFLLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNqRSxJQUFJLEtBQUssR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFFLGVBQWUsQ0FBQyxDQUFBO2dCQUNyQyxLQUFLLENBQUMsY0FBYyxDQUFDLEdBQUcsRUFBQyxFQUFFLENBQUMsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO2FBQ2hCO1NBQ0o7UUFFRCxJQUFJLEtBQUssQ0FBQyxNQUFNLElBQUksS0FBSyxFQUFFO1lBQ3ZCLElBQUksc0JBQU0sQ0FBQyxhQUFhLElBQUssSUFBSSxFQUFFO2dCQUMvQixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUksSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUUsQ0FBQzthQUdoRjtpQkFBTTtnQkFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxFQUFJO29CQUN2RyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztpQkFDM0U7YUFFSjtTQUNQO0lBR0wsQ0FBQztJQTdFZ0IsT0FBTztRQUQzQixPQUFPO09BQ2EsT0FBTyxDQThFM0I7SUFBRCxjQUFDO0NBOUVELEFBOEVDLENBOUVvQyxFQUFFLENBQUMsU0FBUyxHQThFaEQ7a0JBOUVvQixPQUFPIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbCAtIDAwMlwiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzMgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgICBjb3VudCA9IDA7XHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcblxyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMuc2NoZWR1bGUoKCkgPT4ge1xyXG4gICAgICAgICAgICBnbG9iYWwubWluaW9uX2F0dGFjayA9IHRydWU7XHJcbiAgICAgICAgfSwxKTtcclxuXHJcbiAgICAgICAgdGhpcy5taW5pb25feCA9IHRoaXMubm9kZS5wb3NpdGlvbi54O1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gdHJ1ZSkge1xyXG4gICAgICAgICAgICBnbG9iYWwubWFpbl9ocCAtPSA1O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItNVwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgXHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgIGdsb2JhbC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYmFsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9iYWwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgXHJcbiAgICAgICAgbGV0IG5vZGUxID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL+Wwj+msvDJcIik7XHJcbiAgICAgICAgaWYgKHRoaXMuY291bnQgPT0gMCkge1xyXG4gICAgICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICwgbm9kZTEucG9zaXRpb24ueSk7ICAgXHJcbiAgICAgICAgICAgICAgICBsZXQgbm9kZTIgPSBjYy5maW5kIChcIkNhbnZhcy9iai9iL2FcIilcclxuICAgICAgICAgICAgICAgIG5vZGUyLnNldENvbnRlbnRTaXplKDQwMCwyNik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLmNvdW50Kys7ICAgICAgICBcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgIGlmIChnbG9iYWwubWluaW9uX2F0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54ID49ICB0aGlzLm1pbmlvbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSAgdGhpcy5taW5pb25feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgXHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level7/right1 - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'b6037q6UWtHgL4wgdVXFDX5', 'right1 - 001');
// 3.16小游戏/command_TypeScript/level7/right1 - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___003_1 = require("./global - 003");
var enemy_1 = /** @class */ (function (_super) {
    __extends(enemy_1, _super);
    function enemy_1() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    enemy_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_1.prototype.start = function () {
        this.schedule(function () {
            global___003_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_1.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___003_1.default.minion_attack == true) {
            global___003_1.default.main_hp -= 20;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-20";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_1.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        global___003_1.default.minion_attack = false;
        if (global___003_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___003_1.default.main_hp, 19);
        }
    };
    enemy_1.prototype.update = function (dt) {
        if (global___003_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    enemy_1 = __decorate([
        ccclass
    ], enemy_1);
    return enemy_1;
}(cc.Component));
exports.default = enemy_1;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDdcXHJpZ2h0MSAtIDAwMS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQywrQ0FBbUM7QUFLbkM7SUFBcUMsMkJBQVk7SUFBakQ7O0lBNERBLENBQUM7SUF2REcsd0JBQU0sR0FBTjtRQUNJLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsbUJBQW1CLEVBQUUsQ0FBQztRQUNoRCxPQUFPLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQztJQUMzQixDQUFDO0lBR0QsdUJBQUssR0FBTDtRQUNJLElBQUksQ0FBQyxRQUFRLENBQUM7WUFDVixzQkFBTSxDQUFDLGFBQWEsR0FBRyxJQUFJLENBQUM7UUFDaEMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBRUwsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQUVELGtDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUN2QixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sR0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDekIsSUFBRyxzQkFBTSxDQUFDLGFBQWEsSUFBSSxJQUFJLEVBQUU7WUFDN0Isc0JBQU0sQ0FBQyxPQUFPLElBQUksRUFBRSxDQUFDO1lBQ3JCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNwRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzdDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN2RCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1NBQzlDO0lBRUwsQ0FBQztJQUVELGlDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNsQixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBR2Ysc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzdCLElBQUksc0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxzQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksc0JBQU0sQ0FBQyxhQUFhLElBQUssSUFBSSxFQUFFO1lBQy9CLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBSSxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBRSxDQUFDO1NBR2hGO2FBQU07WUFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxFQUFJO2dCQUN2RyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMzRTtTQUVKO0lBQ1IsQ0FBQztJQTNEZ0IsT0FBTztRQUQzQixPQUFPO09BQ2EsT0FBTyxDQTREM0I7SUFBRCxjQUFDO0NBNURELEFBNERDLENBNURvQyxFQUFFLENBQUMsU0FBUyxHQTREaEQ7a0JBNURvQixPQUFPIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbCAtIDAwM1wiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgIGlmKGdsb2JhbC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1haW5faHAgLT0gMjA7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi0yMFwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICAgICBjYy5sb2coXCLnorDmkp7nu5PmnZ9cIik7XHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgIGlmKCBnbG9iYWwubWFpbl9ocCA8PSAwKSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBsZXQgbG9zZSA9IGNjLmZpbmQoXCJDYW52YXMvYmovZmFpbFwiKTtcclxuICAgICAgICBsb3NlLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoIGdsb2JhbC5tYWluX2hwLCAxOSk7XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG5cclxuICAgICAgICBpZiAoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgXHJcbiAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gIHRoaXMubWluaW9uX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9ICB0aGlzLm1pbmlvbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIH1cclxuICAgIH1cclxufVxyXG4iXX0=
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level7/global - 003.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '47949gh4R5B/JPwE+6tmnpw', 'global - 003');
// 3.16小游戏/command_TypeScript/level7/global - 003.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global = /** @class */ (function (_super) {
    __extends(global, _super);
    function global() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    global_1 = global;
    global.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    global.prototype.onEventStart = function () {
        global_1.attack = true;
    };
    global.prototype.update = function (dt) {
        cc.log("min1, min2, min3 " + global_1.minion1_hp + " " + global_1.minion3_hp + " " + global_1.minion3_hp);
    };
    var global_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    global.attack = false;
    global.minion_attack = false;
    global.main_hp = 300;
    global.minion1_hp = 600;
    global.minion2_hp = 300;
    global.minion3_hp = 300;
    __decorate([
        property(cc.Label)
    ], global.prototype, "label", void 0);
    __decorate([
        property
    ], global.prototype, "text", void 0);
    global = global_1 = __decorate([
        ccclass
    ], global);
    return global;
}(cc.Component));
exports.default = global;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDdcXGdsb2JhbCAtIDAwMy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUFvQywwQkFBWTtJQUFoRDtRQUFBLHFFQThCQztRQTNCRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBd0IzQixDQUFDO2VBOUJvQixNQUFNO0lBb0J2QixzQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEQsQ0FBQztJQUNELDZCQUFZLEdBQVo7UUFDSSxRQUFNLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztJQUV6QixDQUFDO0lBQ0QsdUJBQU0sR0FBTixVQUFRLEVBQUU7UUFDTixFQUFFLENBQUMsR0FBRyxDQUFDLG1CQUFtQixHQUFDLFFBQU0sQ0FBQyxVQUFVLEdBQUMsR0FBRyxHQUFDLFFBQU0sQ0FBQyxVQUFVLEdBQUMsR0FBRyxHQUFDLFFBQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQTtJQUM3RixDQUFDOztJQXJCRCx3QkFBd0I7SUFFeEIsZUFBZTtJQUVGLGFBQU0sR0FBWSxLQUFLLENBQUM7SUFDeEIsb0JBQWEsR0FBWSxLQUFLLENBQUM7SUFDL0IsY0FBTyxHQUFXLEdBQUcsQ0FBQztJQUN0QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQUN6QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQUN6QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQWR0QztRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO3lDQUNJO0lBR3ZCO1FBREMsUUFBUTt3Q0FDYztJQU5OLE1BQU07UUFEMUIsT0FBTztPQUNhLE1BQU0sQ0E4QjFCO0lBQUQsYUFBQztDQTlCRCxBQThCQyxDQTlCbUMsRUFBRSxDQUFDLFNBQVMsR0E4Qi9DO2tCQTlCb0IsTUFBTSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgZ2xvYmFsIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuXHJcbiAgICBAcHJvcGVydHkoY2MuTGFiZWwpXHJcbiAgICBsYWJlbDogY2MuTGFiZWwgPSBudWxsO1xyXG5cclxuICAgIEBwcm9wZXJ0eVxyXG4gICAgdGV4dDogc3RyaW5nID0gJ2hlbGxvJztcclxuXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuXHJcbiAgICAvLyBvbkxvYWQgKCkge31cclxuICAgXHJcbiAgIHB1YmxpYyBzdGF0aWMgYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uX2F0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1haW5faHA6IG51bWJlciA9IDMwMDtcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24xX2hwOiBudW1iZXIgPSA2MDA7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMl9ocDogbnVtYmVyID0gMzAwO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjNfaHA6IG51bWJlciA9IDMwMDtcclxuICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMubm9kZS5vbigndG91Y2hzdGFydCcsIHRoaXMub25FdmVudFN0YXJ0LCB0aGlzKTtcclxuICAgIH1cclxuICAgIG9uRXZlbnRTdGFydCgpIHtcclxuICAgICAgICBnbG9iYWwuYXR0YWNrID0gdHJ1ZTtcclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgdXBkYXRlIChkdCkge1xyXG4gICAgICAgIGNjLmxvZyhcIm1pbjEsIG1pbjIsIG1pbjMgXCIrZ2xvYmFsLm1pbmlvbjFfaHArXCIgXCIrZ2xvYmFsLm1pbmlvbjNfaHArXCIgXCIrZ2xvYmFsLm1pbmlvbjNfaHApXHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level7/right2 - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'f7c35dWhelG2Z/xjBxSJZru', 'right2 - 001');
// 3.16小游戏/command_TypeScript/level7/right2 - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___003_1 = require("./global - 003");
var enemy_2 = /** @class */ (function (_super) {
    __extends(enemy_2, _super);
    function enemy_2() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_2.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_2.prototype.start = function () {
        this.schedule(function () {
            global___003_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_2.prototype.onCollisionEnter = function (other, self) {
        if (global___003_1.default.minion_attack == true) {
            global___003_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_2.prototype.onCollisionExit = function (other) {
        global___003_1.default.minion_attack = false;
        if (global___003_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___003_1.default.main_hp, 19);
        }
    };
    enemy_2.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼");
        if (this.count == 0) {
            if (node1.active == false) {
                var node2 = cc.find("Canvas/bj/b/a");
                node2.active = true;
                node2.setContentSize(200, 26);
                this.node.setPosition(this.node.position.x, node1.position.y);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (global___003_1.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_2 = __decorate([
        ccclass
    ], enemy_2);
    return enemy_2;
}(cc.Component));
exports.default = enemy_2;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDdcXHJpZ2h0MiAtIDAwMS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQywrQ0FBbUM7QUFLbkM7SUFBcUMsMkJBQVk7SUFBakQ7UUFBQSxxRUE4RUM7UUF6RUksV0FBSyxHQUFHLENBQUMsQ0FBQzs7SUF5RWYsQ0FBQztJQXhFRyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBRTNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLHNCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsa0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBRXZCLElBQUcsc0JBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLHNCQUFNLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUM1QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztJQUdMLENBQUM7SUFFRCxpQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFJakIsc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzlCLElBQUksc0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxzQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsbUJBQW1CLENBQUMsQ0FBQztRQUN6QyxJQUFJLElBQUksQ0FBQyxLQUFLLElBQUksQ0FBQyxFQUFFO1lBQ2pCLElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7Z0JBQ3ZCLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUUsZUFBZSxDQUFDLENBQUE7Z0JBQ3JDLEtBQUssQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO2dCQUNwQixLQUFLLENBQUMsY0FBYyxDQUFDLEdBQUcsRUFBQyxFQUFFLENBQUMsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFHLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2pFLElBQUksQ0FBQyxLQUFLLEVBQUUsQ0FBQzthQUNoQjtTQUNKO1FBRUQsSUFBSSxLQUFLLENBQUMsTUFBTSxJQUFJLEtBQUssRUFBRTtZQUN2QixJQUFJLHNCQUFNLENBQUMsYUFBYSxJQUFLLElBQUksRUFBRTtnQkFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7YUFHaEY7aUJBQU07Z0JBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsRUFBSTtvQkFDdkcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7aUJBQzNFO2FBRUo7U0FDUDtJQUdMLENBQUM7SUE3RWdCLE9BQU87UUFEM0IsT0FBTztPQUNhLE9BQU8sQ0E4RTNCO0lBQUQsY0FBQztDQTlFRCxBQThFQyxDQTlFb0MsRUFBRSxDQUFDLFNBQVMsR0E4RWhEO2tCQTlFb0IsT0FBTyIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIlxyXG4vLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuaW1wb3J0IGdsb2JhbCBmcm9tIFwiLi9nbG9iYWwgLSAwMDNcIlxyXG5cclxuXHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBlbmVteV8yIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuICAgXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuICAgIG1pbmlvbl94O1xyXG4gICAgXHJcbiAgICAgY291bnQgPSAwO1xyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG5cclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgIFxyXG4gICAgICAgIGlmKGdsb2JhbC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1haW5faHAgLT0gNTtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTVcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgXHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgIGdsb2JhbC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYmFsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9iYWwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgXHJcbiAgICAgICAgbGV0IG5vZGUxID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL+Wwj+msvFwiKTtcclxuICAgICAgICBpZiAodGhpcy5jb3VudCA9PSAwKSB7XHJcbiAgICAgICAgICAgIGlmIChub2RlMS5hY3RpdmUgPT0gZmFsc2UpIHtcclxuICAgICAgICAgICAgICAgIGxldCBub2RlMiA9IGNjLmZpbmQgKFwiQ2FudmFzL2JqL2IvYVwiKVxyXG4gICAgICAgICAgICAgICAgbm9kZTIuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIG5vZGUyLnNldENvbnRlbnRTaXplKDIwMCwyNik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24oICB0aGlzLm5vZGUucG9zaXRpb24ueCAsIG5vZGUxLnBvc2l0aW9uLnkpOyAgIFxyXG4gICAgICAgICAgICAgICAgdGhpcy5jb3VudCsrOyAgICAgICAgXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKG5vZGUxLmFjdGl2ZSA9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICBpZiAoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSAgdGhpcy5taW5pb25feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPD0gIHRoaXMubWluaW9uX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIFxyXG4gICAgfVxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level7/right3 - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'f0e01u1aPhDp4Dw+P9f7QGg', 'right3 - 001');
// 3.16小游戏/command_TypeScript/level7/right3 - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___003_1 = require("./global - 003");
var enemy_3 = /** @class */ (function (_super) {
    __extends(enemy_3, _super);
    function enemy_3() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_3.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_3.prototype.start = function () {
        this.schedule(function () {
            global___003_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_3.prototype.onCollisionEnter = function (other, self) {
        if (global___003_1.default.minion_attack == true) {
            global___003_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_3.prototype.onCollisionExit = function (other) {
        global___003_1.default.minion_attack = false;
        if (global___003_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___003_1.default.main_hp, 19);
        }
    };
    enemy_3.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼2");
        if (this.count == 0) {
            if (node1.active == false) {
                this.node.setPosition(this.node.position.x, node1.position.y);
                var node2 = cc.find("Canvas/bj/b/a");
                node2.setContentSize(400, 26);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (global___003_1.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_3 = __decorate([
        ccclass
    ], enemy_3);
    return enemy_3;
}(cc.Component));
exports.default = enemy_3;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDdcXHJpZ2h0MyAtIDAwMS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQywrQ0FBbUM7QUFLbkM7SUFBcUMsMkJBQVk7SUFBakQ7UUFBQSxxRUE4RUM7UUF6RUksV0FBSyxHQUFHLENBQUMsQ0FBQzs7SUF5RWYsQ0FBQztJQXhFRyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBRTNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLHNCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsa0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBRXZCLElBQUcsc0JBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLHNCQUFNLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUM1QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztJQUdMLENBQUM7SUFFRCxpQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFJakIsc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzlCLElBQUksc0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxzQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsb0JBQW9CLENBQUMsQ0FBQztRQUMxQyxJQUFJLElBQUksQ0FBQyxLQUFLLElBQUksQ0FBQyxFQUFFO1lBQ2pCLElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7Z0JBRXZCLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsRUFBRyxLQUFLLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNqRSxJQUFJLEtBQUssR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFFLGVBQWUsQ0FBQyxDQUFBO2dCQUNyQyxLQUFLLENBQUMsY0FBYyxDQUFDLEdBQUcsRUFBQyxFQUFFLENBQUMsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO2FBQ2hCO1NBQ0o7UUFFRCxJQUFJLEtBQUssQ0FBQyxNQUFNLElBQUksS0FBSyxFQUFFO1lBQ3ZCLElBQUksc0JBQU0sQ0FBQyxhQUFhLElBQUssSUFBSSxFQUFFO2dCQUMvQixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUksSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUUsQ0FBQzthQUdoRjtpQkFBTTtnQkFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxFQUFJO29CQUN2RyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztpQkFDM0U7YUFFSjtTQUNQO0lBR0wsQ0FBQztJQTdFZ0IsT0FBTztRQUQzQixPQUFPO09BQ2EsT0FBTyxDQThFM0I7SUFBRCxjQUFDO0NBOUVELEFBOEVDLENBOUVvQyxFQUFFLENBQUMsU0FBUyxHQThFaEQ7a0JBOUVvQixPQUFPIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbCAtIDAwM1wiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzMgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgICBjb3VudCA9IDA7XHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcblxyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMuc2NoZWR1bGUoKCkgPT4ge1xyXG4gICAgICAgICAgICBnbG9iYWwubWluaW9uX2F0dGFjayA9IHRydWU7XHJcbiAgICAgICAgfSwxKTtcclxuXHJcbiAgICAgICAgdGhpcy5taW5pb25feCA9IHRoaXMubm9kZS5wb3NpdGlvbi54O1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gdHJ1ZSkge1xyXG4gICAgICAgICAgICBnbG9iYWwubWFpbl9ocCAtPSA1O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItNVwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgXHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgIGdsb2JhbC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYmFsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9iYWwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgXHJcbiAgICAgICAgbGV0IG5vZGUxID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL+Wwj+msvDJcIik7XHJcbiAgICAgICAgaWYgKHRoaXMuY291bnQgPT0gMCkge1xyXG4gICAgICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICwgbm9kZTEucG9zaXRpb24ueSk7ICAgXHJcbiAgICAgICAgICAgICAgICBsZXQgbm9kZTIgPSBjYy5maW5kIChcIkNhbnZhcy9iai9iL2FcIilcclxuICAgICAgICAgICAgICAgIG5vZGUyLnNldENvbnRlbnRTaXplKDQwMCwyNik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLmNvdW50Kys7ICAgICAgICBcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgIGlmIChnbG9iYWwubWluaW9uX2F0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54ID49ICB0aGlzLm1pbmlvbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSAgdGhpcy5taW5pb25feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgXHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level8/right - 002.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'e5effeAEFhCT5MxbTLHTyWa', 'right - 002');
// 3.16小游戏/command_TypeScript/level8/right - 002.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___004_1 = require("./global - 004");
var right = /** @class */ (function (_super) {
    __extends(right, _super);
    function right() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    right.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    right.prototype.start = function () {
        this.schedule(function () {
            global___004_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    right.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___004_1.default.minion_attack == true) {
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-100";
            var damage2 = cc.find("Canvas/bj/kuan/main_damage");
            damage2.getComponent(cc.Label).string = "";
            global___004_1.default.main_hp -= 40;
        }
    };
    right.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        global___004_1.default.minion_attack = false;
        if (global___004_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___004_1.default.main_hp, 19);
        }
    };
    right.prototype.update = function (dt) {
        if (global___004_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    right = __decorate([
        ccclass
    ], right);
    return right;
}(cc.Component));
exports.default = right;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDhcXHJpZ2h0IC0gMDAyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLCtDQUFtQztBQUtuQztJQUFtQyx5QkFBWTtJQUEvQzs7SUE2REEsQ0FBQztJQXhERyxzQkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBQzNCLENBQUM7SUFHRCxxQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLHNCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsZ0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBQ3ZCLEVBQUUsQ0FBQyxHQUFHLENBQUMsTUFBTSxHQUFDLEtBQUssQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUN6QixJQUFHLHNCQUFNLENBQUMsYUFBYSxJQUFJLElBQUksRUFBRTtZQUM3QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQztZQUM5QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDRCQUE0QixDQUFDLENBQUM7WUFDcEQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztZQUMzQyxzQkFBTSxDQUFDLE9BQU8sSUFBSSxFQUFFLENBQUM7U0FDeEI7SUFHTCxDQUFDO0lBRUQsK0JBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2xCLEVBQUUsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7UUFHZixzQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7UUFDN0IsSUFBSSxzQkFBTSxDQUFDLE9BQU8sSUFBSSxDQUFDLEVBQUU7WUFDeEIsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLElBQUksSUFBSSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsZ0JBQWdCLENBQUMsQ0FBQztZQUNyQyxJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztTQUNuQjthQUFNO1lBQ04sS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsY0FBYyxDQUFFLHNCQUFNLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUdELHNCQUFNLEdBQU4sVUFBUSxFQUFFO1FBRU4sSUFBSSxzQkFBTSxDQUFDLGFBQWEsSUFBSyxJQUFJLEVBQUU7WUFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7U0FHaEY7YUFBTTtZQUNILElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLEVBQUk7Z0JBQ3ZHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzNFO1NBRUo7SUFDUixDQUFDO0lBNURnQixLQUFLO1FBRHpCLE9BQU87T0FDYSxLQUFLLENBNkR6QjtJQUFELFlBQUM7Q0E3REQsQUE2REMsQ0E3RGtDLEVBQUUsQ0FBQyxTQUFTLEdBNkQ5QztrQkE3RG9CLEtBQUsiLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyJcclxuLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcbmltcG9ydCBnbG9hYmwgZnJvbSBcIi4vZ2xvYmFsIC0gMDA0XCJcclxuXHJcblxyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgcmlnaHQgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYWJsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgIGlmKGdsb2FibC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTEwMFwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgICAgIGdsb2FibC5tYWluX2hwIC09IDQwO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgICAgY2MubG9nKFwi56Kw5pKe57uT5p2fXCIpO1xyXG4gICAgICAgIFxyXG4gICAgICBcclxuICAgICAgIGdsb2FibC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYWJsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9hYmwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuXHJcbiAgICAgICAgaWYgKGdsb2FibC5taW5pb25fYXR0YWNrID09ICB0cnVlKSB7XHJcbiAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIFxyXG4gICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54ID49ICB0aGlzLm1pbmlvbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSAgdGhpcy5taW5pb25feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCArIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICBcclxuICAgICAgICAgICB9XHJcbiAgICB9XHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level8/global - 004.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '7a776aCQs1Ew4XgHctTZOLE', 'global - 004');
// 3.16小游戏/command_TypeScript/level8/global - 004.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var gloabl = /** @class */ (function (_super) {
    __extends(gloabl, _super);
    function gloabl() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    gloabl_1 = gloabl;
    gloabl.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    gloabl.prototype.onEventStart = function () {
        cc.log("click");
        gloabl_1.attack = true;
    };
    var gloabl_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    gloabl.attack = false;
    gloabl.minion_attack = false;
    gloabl.main_hp = 500;
    gloabl.minion1_hp = 20000;
    gloabl.minion2_hp = 163;
    gloabl.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], gloabl.prototype, "label", void 0);
    __decorate([
        property
    ], gloabl.prototype, "text", void 0);
    gloabl = gloabl_1 = __decorate([
        ccclass
    ], gloabl);
    return gloabl;
}(cc.Component));
exports.default = gloabl;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDhcXGdsb2JhbCAtIDAwNC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUFvQywwQkFBWTtJQUFoRDtRQUFBLHFFQTRCQztRQXpCRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBc0IzQixDQUFDO2VBNUJvQixNQUFNO0lBb0J2QixzQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEQsQ0FBQztJQUNELDZCQUFZLEdBQVo7UUFDSSxFQUFFLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ2hCLFFBQU0sQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO0lBQ3pCLENBQUM7O0lBbEJELHdCQUF3QjtJQUV4QixlQUFlO0lBRUYsYUFBTSxHQUFZLEtBQUssQ0FBQztJQUN4QixvQkFBYSxHQUFZLEtBQUssQ0FBQztJQUMvQixjQUFPLEdBQVcsR0FBRyxDQUFDO0lBQ3RCLGlCQUFVLEdBQVcsS0FBSyxDQUFDO0lBQzNCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGlCQUFVLEdBQVcsR0FBRyxDQUFDO0lBZHRDO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7eUNBQ0k7SUFHdkI7UUFEQyxRQUFRO3dDQUNjO0lBTk4sTUFBTTtRQUQxQixPQUFPO09BQ2EsTUFBTSxDQTRCMUI7SUFBRCxhQUFDO0NBNUJELEFBNEJDLENBNUJtQyxFQUFFLENBQUMsU0FBUyxHQTRCL0M7a0JBNUJvQixNQUFNIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBnbG9hYmwgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG5cclxuICAgIEBwcm9wZXJ0eShjYy5MYWJlbClcclxuICAgIGxhYmVsOiBjYy5MYWJlbCA9IG51bGw7XHJcblxyXG4gICAgQHByb3BlcnR5XHJcbiAgICB0ZXh0OiBzdHJpbmcgPSAnaGVsbG8nO1xyXG5cclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgXHJcbiAgICAvLyBvbkxvYWQgKCkge31cclxuICAgXHJcbiAgIHB1YmxpYyBzdGF0aWMgYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uX2F0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1haW5faHA6IG51bWJlciA9IDUwMDtcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24xX2hwOiBudW1iZXIgPSAyMDAwMDtcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24yX2hwOiBudW1iZXIgPSAxNjM7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uM19ocDogbnVtYmVyID0gMTYzO1xyXG4gICBcclxuXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgICAgdGhpcy5ub2RlLm9uKCd0b3VjaHN0YXJ0JywgdGhpcy5vbkV2ZW50U3RhcnQsIHRoaXMpO1xyXG4gICAgfVxyXG4gICAgb25FdmVudFN0YXJ0KCkge1xyXG4gICAgICAgIGNjLmxvZyhcImNsaWNrXCIpO1xyXG4gICAgICAgIGdsb2FibC5hdHRhY2sgPSB0cnVlO1xyXG4gICAgfVxyXG4gICAgXHJcbn1cclxuIl19
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level1/mian.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'd4300SSbLBAmqdaWSSlm/Ka', 'mian');
// 3.16小游戏/command_TypeScript/level1/mian.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main = /** @class */ (function (_super) {
    __extends(main, _super);
    function main() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    main_1 = main;
    main.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    main.prototype.onEventStart = function () {
        cc.log("click");
        main_1.attack = true;
    };
    var main_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    main.attack = false;
    main.minion_attack = false;
    main.main_hp = 163;
    main.minion1_hp = 163;
    main.minion2_hp = 163;
    main.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], main.prototype, "label", void 0);
    __decorate([
        property
    ], main.prototype, "text", void 0);
    main = main_1 = __decorate([
        ccclass
    ], main);
    return main;
}(cc.Component));
exports.default = main;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDFcXG1pYW4udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7OztBQUFBLG9CQUFvQjtBQUNwQix3RUFBd0U7QUFDeEUsbUJBQW1CO0FBQ25CLGtGQUFrRjtBQUNsRiw4QkFBOEI7QUFDOUIsa0ZBQWtGOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFFNUUsSUFBQSxLQUFzQixFQUFFLENBQUMsVUFBVSxFQUFsQyxPQUFPLGFBQUEsRUFBRSxRQUFRLGNBQWlCLENBQUM7QUFHMUM7SUFBa0Msd0JBQVk7SUFBOUM7UUFBQSxxRUE0QkM7UUF6QkcsV0FBSyxHQUFhLElBQUksQ0FBQztRQUd2QixVQUFJLEdBQVcsT0FBTyxDQUFDOztJQXNCM0IsQ0FBQzthQTVCb0IsSUFBSTtJQW9CckIsb0JBQUssR0FBTDtRQUNJLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLFlBQVksRUFBRSxJQUFJLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3hELENBQUM7SUFDRCwyQkFBWSxHQUFaO1FBQ0ksRUFBRSxDQUFDLEdBQUcsQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNoQixNQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztJQUN2QixDQUFDOztJQWxCRCx3QkFBd0I7SUFFeEIsZUFBZTtJQUVGLFdBQU0sR0FBWSxLQUFLLENBQUM7SUFDeEIsa0JBQWEsR0FBWSxLQUFLLENBQUM7SUFDL0IsWUFBTyxHQUFXLEdBQUcsQ0FBQztJQUN0QixlQUFVLEdBQVcsR0FBRyxDQUFDO0lBQ3pCLGVBQVUsR0FBVyxHQUFHLENBQUM7SUFDekIsZUFBVSxHQUFXLEdBQUcsQ0FBQztJQWR0QztRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO3VDQUNJO0lBR3ZCO1FBREMsUUFBUTtzQ0FDYztJQU5OLElBQUk7UUFEeEIsT0FBTztPQUNhLElBQUksQ0E0QnhCO0lBQUQsV0FBQztDQTVCRCxBQTRCQyxDQTVCaUMsRUFBRSxDQUFDLFNBQVMsR0E0QjdDO2tCQTVCb0IsSUFBSSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgbWFpbiBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcbiAgICBcclxuICAgIC8vIG9uTG9hZCAoKSB7fVxyXG4gICBcclxuICAgcHVibGljIHN0YXRpYyBhdHRhY2s6IGJvb2xlYW4gPSBmYWxzZTtcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb25fYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWFpbl9ocDogbnVtYmVyID0gMTYzO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjFfaHA6IG51bWJlciA9IDE2MztcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24yX2hwOiBudW1iZXIgPSAxNjM7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uM19ocDogbnVtYmVyID0gMTYzO1xyXG4gICBcclxuXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgICAgdGhpcy5ub2RlLm9uKCd0b3VjaHN0YXJ0JywgdGhpcy5vbkV2ZW50U3RhcnQsIHRoaXMpO1xyXG4gICAgfVxyXG4gICAgb25FdmVudFN0YXJ0KCkge1xyXG4gICAgICAgIGNjLmxvZyhcImNsaWNrXCIpO1xyXG4gICAgICAgIG1haW4uYXR0YWNrID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------

                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/exit.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '9acdd4KJ5hB76Neunaa5p3S', 'exit');
// 3.16小游戏/command_TypeScript/exit.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var NewClass = /** @class */ (function (_super) {
    __extends(NewClass, _super);
    function NewClass() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
        // update (dt) {}
    }
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    NewClass.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    NewClass.prototype.onEventStart = function () {
        cc.game.end();
    };
    __decorate([
        property(cc.Label)
    ], NewClass.prototype, "label", void 0);
    __decorate([
        property
    ], NewClass.prototype, "text", void 0);
    NewClass = __decorate([
        ccclass
    ], NewClass);
    return NewClass;
}(cc.Component));
exports.default = NewClass;

cc._RF.pop();
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxleGl0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQXNDLDRCQUFZO0lBQWxEO1FBQUEscUVBb0JDO1FBakJHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7UUFhdkIsaUJBQWlCO0lBQ3JCLENBQUM7SUFaRyx3QkFBd0I7SUFFeEIsZUFBZTtJQUNmLHdCQUFLLEdBQUw7UUFDSSxJQUFJLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLFlBQVksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUN4RCxDQUFDO0lBQ0QsK0JBQVksR0FBWjtRQUNJLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDbEIsQ0FBQztJQWJEO1FBREMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUM7MkNBQ0k7SUFHdkI7UUFEQyxRQUFROzBDQUNjO0lBTk4sUUFBUTtRQUQ1QixPQUFPO09BQ2EsUUFBUSxDQW9CNUI7SUFBRCxlQUFDO0NBcEJELEFBb0JDLENBcEJxQyxFQUFFLENBQUMsU0FBUyxHQW9CakQ7a0JBcEJvQixRQUFRIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBOZXdDbGFzcyBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcblxyXG4gICAgLy8gb25Mb2FkICgpIHt9XHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgICAgdGhpcy5ub2RlLm9uKCd0b3VjaHN0YXJ0JywgdGhpcy5vbkV2ZW50U3RhcnQsIHRoaXMpO1xyXG4gICAgfVxyXG4gICAgb25FdmVudFN0YXJ0KCkge1xyXG4gICAgICAgIGNjLmdhbWUuZW5kKCk7XHJcbiAgICB9XHJcbiAgICBcclxuXHJcbiAgICAvLyB1cGRhdGUgKGR0KSB7fVxyXG59XHJcbiJdfQ==
//------QC-SOURCE-SPLIT------
