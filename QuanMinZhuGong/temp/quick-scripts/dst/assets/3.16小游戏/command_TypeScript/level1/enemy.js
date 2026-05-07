
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