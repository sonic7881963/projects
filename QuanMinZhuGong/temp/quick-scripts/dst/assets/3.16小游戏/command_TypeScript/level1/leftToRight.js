
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDFcXGxlZnRUb1JpZ2h0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBLG9CQUFvQjtBQUNwQix3RUFBd0U7QUFDeEUsbUJBQW1CO0FBQ25CLGtGQUFrRjtBQUNsRiw4QkFBOEI7QUFDOUIsa0ZBQWtGO0FBQ2xGLCtCQUF5QjtBQUtuQixJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUF5QywrQkFBWTtJQUFyRDtRQUFBLHFFQXNGQztRQW5GRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBZ0YzQixDQUFDO29CQXRGb0IsV0FBVztJQVk1Qix3QkFBd0I7SUFFeEIsNEJBQU0sR0FBTjtRQUNJLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsbUJBQW1CLEVBQUUsQ0FBQztRQUNoRCxPQUFPLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQztJQUMzQixDQUFDO0lBR0QsU0FBUztJQUNULHNDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUN2QixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sR0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLENBQUM7UUFFekIsSUFBRyxjQUFJLENBQUMsTUFBTSxJQUFJLElBQUksRUFBQztZQUNuQixjQUFJLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQztZQUN0QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUM3QyxJQUFJLE9BQU8sR0FBRyxFQUFF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bnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5pbXBvcnQgbWFpbiBmcm9tIFwiLi9taWFuXCJcclxuaW1wb3J0IGVuZW15IGZyb20gJy4vZW5lbXknO1xyXG5cclxuXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGxlZnRUb1JpZ2h0IGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuXHJcbiAgICBAcHJvcGVydHkoY2MuTGFiZWwpXHJcbiAgICBsYWJlbDogY2MuTGFiZWwgPSBudWxsO1xyXG5cclxuICAgIEBwcm9wZXJ0eVxyXG4gICAgdGV4dDogc3RyaW5nID0gJ2hlbGxvJztcclxuICAgIG1haW5feDogbnVtYmVyO1xyXG4gICAgXHJcbiAgICBwdWJsaWMgc3RhdGljIGN1cnJlbnRfeDogbnVtYmVyO1xyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3k6IG51bWJlcjtcclxuXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuXHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcbiAgICB9XHJcbiAgICBcclxuXHJcbiAgICAvL+S6p+eUn+eisOaSnuS8muiwg+eUqFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYobWFpbi5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIG1haW4ubWluaW9uMV9ocCAtPSAzMDtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItMzBcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgbWFpbi5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgb25Db2xsaXNpb25TdGF5KG90aGVyKSB7XHJcbiAgICAgICAgbWFpbi5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBtYWluLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgIH1cclxuXHJcbiAgICBvbkNvbGxpc2lvbkV4aXQob3RoZXIpIHtcclxuICAgICAgIGNjLmxvZyhcIueisOaSnue7k+adn1wiKTtcclxuICAgICAgIFxyXG4gICAgICAgXHJcbiAgICAgIFxyXG4gICAgICBcclxuICAgICAgIGlmKG1haW4ubWluaW9uMV9ocCA8PSAwKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBtYWluLmF0dGFjaz0gZmFsc2U7XHJcbiAgICAgICAgbWFpbi5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICAgbWFpbi5tYWluX2hwPSAxNjM7XHJcbiAgICAgICAgbWFpbi5taW5pb24xX2hwPSAxNjM7XHJcbiAgICAgICAgbWFpbi5taW5pb24yX2hwPSAxNjM7XHJcbiAgICAgICAgbWFpbi5taW5pb24zX2hwID0gMTYzO1xyXG4gICAgICAgIGNjLmRpcmVjdG9yLmxvYWRTY2VuZShcImZpZ2h0MlwiKTtcclxuICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShtYWluLm1pbmlvbjFfaHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgIFxyXG4gICAgdGhpcy5tYWluX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgIFxyXG4gICAgIFxyXG4gICAgfVxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBpZiAobWFpbi5hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICBcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSB0aGlzLm1haW5feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gdGhpcy5tYWluX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggLSAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBsZWZ0VG9SaWdodC5jdXJyZW50X3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgICAgICBsZWZ0VG9SaWdodC5jdXJyZW50X3kgPSB0aGlzLm5vZGUucG9zaXRpb24ueTtcclxuICAgIH1cclxuICAgIFxyXG59XHJcblxyXG5cclxuIl19